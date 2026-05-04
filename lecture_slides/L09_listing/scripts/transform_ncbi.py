"""
Runs inside the Apptainer container (python:3.11-slim).
Reads raw_GENE_DS.json from /data, writes summary_GENE_DS.json.
Receives DATA_DIR, GENE, DS via environment variables.
"""
import json
import os

data_dir = os.environ["DATA_DIR"]
gene = os.environ["GENE"]
ds = os.environ["DS"]

with open(f"{data_dir}/raw_{gene}_{ds}.json") as f:
    data = json.load(f)

summary = {
    "gene": data["gene"],
    "execution_date": data["execution_date"],
    "total_hits": data["count"],
    "sampled_ids": data["ids"],
    "id_count": len(data["ids"]),
}

out_path = f"{data_dir}/summary_{gene}_{ds}.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"summary written: {out_path}")
