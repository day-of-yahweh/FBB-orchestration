"""
dag_ncbi_pipeline.py
--------------------
Demonstration Airflow pipeline:
  1. Fetch gene publication IDs from NCBI PubMed (NCBI E-utilities, no auth needed)
  2. Branch: proceed if results found, skip otherwise
  3. Transform raw JSON inside an Apptainer container (SingularityOperator)
  4. Save a human-readable report (PythonOperator)
  5. Trigger a downstream notification DAG

Concepts demonstrated:
  - DAG with params and Jinja templating ({{ ds }}, {{ params.gene }}, {{ ti.xcom_pull(...) }})
  - PythonOperator, BashOperator, BranchPythonOperator, SingularityOperator, TriggerDagRunOperator
  - XCom push/pull
  - TriggerRule on a join task after branching
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.singularity.operators.singularity import SingularityOperator
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow"))
DATA_DIR = os.path.join(AIRFLOW_HOME, "data", "ncbi")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _fetch_publications(gene: str, max_results: int, ds: str, **context) -> int:
    """
    Call NCBI esearch to find PubMed IDs for *gene* published in the
    year of the DAG execution date. Pushes a result dict to XCom and
    writes raw JSON to DATA_DIR.

    Returns the total hit count (used as the task return value, also
    automatically pushed to XCom under key 'return_value').
    """
    import requests  # provider import inside callable to keep module-level clean

    os.makedirs(DATA_DIR, exist_ok=True)

    year = ds[:4]  # Jinja passes 'ds' as YYYY-MM-DD string
    params = {
        "db": "pubmed",
        "term": f"{gene}[Gene] AND {year}[PDAT]",
        "retmax": int(max_results),
        "retmode": "json",
        "usehistory": "y",
    }
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()

    esresult = resp.json()["esearchresult"]
    payload = {
        "gene": gene,
        "execution_date": ds,
        "count": int(esresult["count"]),
        "ids": esresult["idlist"],
        "webenv": esresult.get("webenv", ""),
        "query_key": esresult.get("querykey", ""),
    }

    raw_path = os.path.join(DATA_DIR, f"raw_{gene}_{ds}.json")
    with open(raw_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    ti = context["ti"]
    ti.xcom_push(key="ncbi_payload", value=payload)
    ti.xcom_push(key="raw_path", value=raw_path)

    print(f"[fetch] gene={gene} year={year} hits={payload['count']} ids={payload['ids']}")
    return payload["count"]


def _branch_on_count(**context) -> str:
    """
    Pull the result count from XCom and decide which branch to take.
    Returns the task_id of the next task to run.
    """
    payload = context["ti"].xcom_pull(task_ids="fetch_publications", key="ncbi_payload")
    count = payload["count"] if payload else 0
    print(f"[branch] count={count}")
    return "transform_with_apptainer" if count > 0 else "log_no_results"


def _save_report(gene: str, ds: str, **context) -> str:
    """
    Pull the summary written by the Apptainer task from DATA_DIR,
    produce a plain-text report, and push the report path to XCom.
    """
    summary_path = os.path.join(DATA_DIR, f"summary_{gene}_{ds}.json")

    # Graceful degradation: if Apptainer step was skipped, create a stub
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)
    else:
        raw_payload = context["ti"].xcom_pull(task_ids="fetch_publications", key="ncbi_payload")
        summary = raw_payload or {"gene": gene, "execution_date": ds, "total_hits": 0, "sampled_ids": []}

    report_path = os.path.join(DATA_DIR, f"report_{gene}_{ds}.txt")
    with open(report_path, "w") as fh:
        fh.write(f"NCBI PubMed Report — {summary.get('gene', gene)} — {ds}\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Total hits in database : {summary.get('total_hits', summary.get('count', 'N/A'))}\n")
        fh.write(f"IDs sampled            : {summary.get('id_count', len(summary.get('sampled_ids', [])))}\n\n")
        for pid in summary.get("sampled_ids", []):
            fh.write(f"  https://pubmed.ncbi.nlm.nih.gov/{pid}/\n")

    context["ti"].xcom_push(key="report_path", value=report_path)
    print(f"[save_report] wrote {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="ncbi_gene_pipeline",
    description="Fetch NCBI PubMed data for a gene, transform, report, and notify",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ncbi", "bioinformatics", "demo"],
    params={
        "gene": "BRCA1",       # gene symbol to query
        "max_results": 20,     # max PubMed IDs to retrieve
    },
    doc_md=__doc__,
) as dag:

    # ------------------------------------------------------------------
    # 1. Fetch — PythonOperator + Jinja templating in op_kwargs
    # ------------------------------------------------------------------
    # {{ ds }}          → logical execution date, e.g. "2025-03-10"
    # {{ params.gene }} → DAG param, overridable at trigger time
    fetch_publications = PythonOperator(
        task_id="fetch_publications",
        python_callable=_fetch_publications,
        op_kwargs={
            "gene": "{{ params.gene }}",
            "max_results": "{{ params.max_results }}",
            "ds": "{{ ds }}",
        },
    )

    # ------------------------------------------------------------------
    # 2. Branch — decide based on XCom'd hit count
    # ------------------------------------------------------------------
    branch_on_count = BranchPythonOperator(
        task_id="branch_on_count",
        python_callable=_branch_on_count,
    )

    # ------------------------------------------------------------------
    # 3a. Transform — SingularityOperator runs python:3.11-slim via Apptainer.
    #     The script is a real file mounted read-only at /scripts — avoids
    #     all shell quoting issues that come with passing -c inline strings.
    #     GENE and DS are injected as environment variables (Jinja-templated).
    # ------------------------------------------------------------------
    transform_with_apptainer = SingularityOperator(
        task_id="transform_with_apptainer",
        image="docker://python:3.11-slim",   # pulled on first use; swap for a local .sif path
        command="python3 /scripts/transform_ncbi.py",
        environment={
            "DATA_DIR": "/data",
            "GENE": "{{ params.gene }}",
            "DS": "{{ ds }}",
        },
        volumes=[
            f"{DATA_DIR}:/data",
            f"{SCRIPTS_DIR}:/scripts:ro",
        ],
    )

    # ------------------------------------------------------------------
    # 3b. Skip branch — BashOperator, Jinja in bash_command
    # ------------------------------------------------------------------
    log_no_results = BashOperator(
        task_id="log_no_results",
        bash_command=(
            "echo 'No PubMed results for gene={{ params.gene }} "
            "on execution_date={{ ds }}. Skipping pipeline.'"
        ),
    )

    # ------------------------------------------------------------------
    # 4. Save report — joins both branches
    #    TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS: runs when at least one
    #    upstream succeeded and none failed (correct after a branch).
    # ------------------------------------------------------------------
    save_report = PythonOperator(
        task_id="save_report",
        python_callable=_save_report,
        op_kwargs={
            "gene": "{{ params.gene }}",
            "ds": "{{ ds }}",
        },
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ------------------------------------------------------------------
    # 5. Trigger downstream notification DAG
    #    Passes report_path via XCom through conf so the notifier knows
    #    where to read from.
    #    TriggerRule.ALL_DONE: fires regardless of save_report outcome.
    # ------------------------------------------------------------------
    trigger_notifier = TriggerDagRunOperator(
        task_id="trigger_notifier",
        trigger_dag_id="ncbi_pipeline_notify",   # must exist separately
        conf={
            "source_dag": "ncbi_gene_pipeline",
            "gene": "{{ params.gene }}",
            "execution_date": "{{ ds }}",
            # XCom pull inside a Jinja expression in conf (supported in 2.x)
            "report_path": "{{ ti.xcom_pull(task_ids='save_report', key='report_path') }}",
        },
        wait_for_completion=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ------------------------------------------------------------------
    # Dependency graph
    #
    #   fetch_publications
    #         │
    #   branch_on_count
    #        / \
    # apptainer log_no_results
    #        \ /
    #    save_report
    #         │
    #   trigger_notifier
    # ------------------------------------------------------------------
    fetch_publications >> branch_on_count >> [transform_with_apptainer, log_no_results]
    transform_with_apptainer >> save_report
    log_no_results >> save_report
    save_report >> trigger_notifier
