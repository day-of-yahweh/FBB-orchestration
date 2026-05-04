"""
dag_ncbi_launcher.py
--------------------
Launcher DAG that conditionally triggers ncbi_gene_pipeline for a
list of genes every Monday morning.

Launch conditions (both must be true):
  1. The previous run of ncbi_gene_pipeline did not end in FAILED state.
  2. Today is a weekday (guard against manual triggers on weekends if
     the schedule is changed).

Concepts demonstrated:
  - BranchPythonOperator with multi-target branch (list of task IDs)
  - Querying DagRun state programmatically
  - TriggerDagRunOperator with conf and wait_for_completion
  - Dynamic task generation inside a DAG (loop over gene list)
  - Jinja in BashOperator
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import DagRun
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.state import DagRunState
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_DAG_ID = "ncbi_gene_pipeline"

# Each entry: (gene_symbol, max_results)
GENE_CONFIGS: list[tuple[str, int]] = [
    ("BRCA1", 50),
    ("TP53",  50),
    ("EGFR",  30),
]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _check_previous_run(**context) -> str | list[str]:
    """
    Inspect the most recent DagRun of TARGET_DAG_ID.
    - If it failed → branch to 'abort_launch'
    - If today is a weekend (safety guard) → branch to 'abort_launch'
    - Otherwise → branch to all trigger tasks (list of task IDs)
    """
    today = context["logical_date"]  # pendulum.DateTime

    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        print(f"[launcher] today={today.date()} is a weekend — aborting.")
        return "abort_launch"

    runs: list[DagRun] = DagRun.find(dag_id=TARGET_DAG_ID, state=DagRunState.FAILED)
    if runs:
        latest_failure = max(runs, key=lambda r: r.execution_date)
        print(f"[launcher] found failed run: {latest_failure.run_id} — aborting.")
        return "abort_launch"

    trigger_task_ids = [f"trigger_{gene.lower()}" for gene, _ in GENE_CONFIGS]
    print(f"[launcher] conditions met — triggering: {trigger_task_ids}")
    return trigger_task_ids


def _summarise_launches(**context) -> None:
    """
    Pull return values from each trigger task via XCom and log a summary.
    (Demonstrates XCom pull from multiple upstream tasks.)
    """
    ti = context["ti"]
    for gene, _ in GENE_CONFIGS:
        result = ti.xcom_pull(task_ids=f"trigger_{gene.lower()}")
        print(f"[summary] trigger_{gene.lower()} → {result}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="ncbi_pipeline_launcher",
    description="Conditionally trigger ncbi_gene_pipeline for multiple genes",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 6 * * 1",   # Every Monday at 06:00
    catchup=False,
    tags=["ncbi", "launcher", "demo"],
    doc_md=__doc__,
) as dag:

    # ------------------------------------------------------------------
    # 1. Guard check + branch decision
    # ------------------------------------------------------------------
    check_conditions = BranchPythonOperator(
        task_id="check_conditions",
        python_callable=_check_previous_run,
    )

    # ------------------------------------------------------------------
    # 2. Abort path — Jinja in bash_command shows execution context
    # ------------------------------------------------------------------
    abort_launch = BashOperator(
        task_id="abort_launch",
        bash_command=(
            "echo 'Launch aborted on {{ ds }} ({{ logical_date }})."
            " Check ncbi_gene_pipeline for failures or weekend guard.'"
        ),
    )

    # ------------------------------------------------------------------
    # 3. Trigger tasks — one per gene, generated dynamically
    #    Each TriggerDagRunOperator passes its own conf dict so the
    #    target DAG receives the right gene symbol at runtime.
    # ------------------------------------------------------------------
    trigger_tasks: list[TriggerDagRunOperator] = []

    for gene, max_results in GENE_CONFIGS:
        task_id = f"trigger_{gene.lower()}"
        t = TriggerDagRunOperator(
            task_id=task_id,
            trigger_dag_id=TARGET_DAG_ID,
            conf={
                "gene": gene,
                "max_results": max_results,
            },
            # Wait for the triggered run to finish before this task completes.
            # With multiple genes this means they run sequentially in the launcher;
            # set to False for fire-and-forget parallel triggering.
            wait_for_completion=True,
            poke_interval=60,
            # Allow the triggered DAG to fail without failing this task —
            # we want the other genes to still be attempted.
            allowed_states=[DagRunState.SUCCESS, DagRunState.FAILED],
            trigger_rule=TriggerRule.ALL_SUCCESS,  # only if check_conditions branched here
        )
        trigger_tasks.append(t)

    # ------------------------------------------------------------------
    # 4. Summary — joins all trigger tasks, runs even if some failed
    # ------------------------------------------------------------------
    summarise_launches = PythonOperator(
        task_id="summarise_launches",
        python_callable=_summarise_launches,
        trigger_rule=TriggerRule.NONE_SKIPPED,
    )

    # ------------------------------------------------------------------
    # Dependency graph
    #
    #   check_conditions
    #      /    |    \  \
    #  abort  trg1  trg2 trg3
    #             \   |  /
    #          summarise_launches
    #
    # abort_launch is NOT connected to summarise_launches so the summary
    # only runs when at least one gene was triggered.
    # ------------------------------------------------------------------
    check_conditions >> abort_launch
    check_conditions >> trigger_tasks
    trigger_tasks >> summarise_launches
