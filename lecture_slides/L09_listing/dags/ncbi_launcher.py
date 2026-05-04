from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import DagRun
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.state import DagRunState


TARGET_DAG_ID = "ncbi_gene_pipeline"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _check_conditions(**context) -> str:
    dag_run = context.get("dag_run")

    # UI-trigger required
    if not dag_run or not dag_run.conf:
        return "abort_launch"

    failed = DagRun.find(
        dag_id=TARGET_DAG_ID,
        state=DagRunState.FAILED,
    )

    if failed:
        return "abort_launch"

    return "extract_payloads"


def _extract_payloads(**context) -> list[dict]:
    conf = context["dag_run"].conf

    genes = conf.get("genes", [])

    if not genes:
        raise ValueError("Missing 'genes' in dag_run.conf")

    return [
        {
            "gene": g["gene"],
            "max_results": g.get("max_results", 20),
        }
        for g in genes
    ]




with DAG(
    dag_id="ncbi_pipeline_launcher2",
    description="Manual launcher for gene pipelines",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ncbi", "launcher"],

    params={
        "genes": Param(
            default=[
                {"gene": "BRCA1", "max_results": 50}
            ],
            type="array",
            items={
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["gene"],
            },
        )
    },
) as dag:

    check_conditions = BranchPythonOperator(
        task_id="check_conditions",
        python_callable=_check_conditions,
    )

    abort_launch = BashOperator(
        task_id="abort_launch",
        bash_command="echo 'Aborted: missing input or failed dependency'",
    )

    extract_payloads = PythonOperator(
        task_id="extract_payloads",
        python_callable=_extract_payloads,
    )

    trigger_gene_pipeline = TriggerDagRunOperator.partial(
        task_id="trigger_gene_pipeline",
        trigger_dag_id=TARGET_DAG_ID,
        wait_for_completion=False,
    ).expand(
        conf=extract_payloads.output
    )
    check_conditions >> abort_launch
    check_conditions >> extract_payloads >> trigger_gene_pipeline
