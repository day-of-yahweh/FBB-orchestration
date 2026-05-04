"""
dag_ncbi_notify.py
------------------
Stub notification DAG triggered by ncbi_gene_pipeline upon completion.
Receives gene, execution_date, and report_path via conf (DagRun.conf).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "email_on_failure": False,
}


def _log_notification(**context) -> None:
    conf = context["dag_run"].conf or {}
    gene = conf.get("gene", "unknown")
    exec_date = conf.get("execution_date", "unknown")
    report_path = conf.get("report_path", "not provided")
    source_dag = conf.get("source_dag", "unknown")

    print(f"[notify] triggered by     : {source_dag}")
    print(f"[notify] gene             : {gene}")
    print(f"[notify] execution_date   : {exec_date}")
    print(f"[notify] report_path      : {report_path}")


with DAG(
    dag_id="ncbi_pipeline_notify",
    description="Receives completion signal from ncbi_gene_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ncbi", "notify", "demo"],
    doc_md=__doc__,
) as dag:

    log_notification = PythonOperator(
        task_id="log_notification",
        python_callable=_log_notification,
    )
