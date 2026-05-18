import requests

BASE_URL = "https://www.ebi.ac.uk/interpro/api"
resp = requests.get(f"{BASE_URL}/entry/interpro", params={"page_size": 5})
entries = resp.json()["results"]

for entry in entries:
    accession = entry["metadata"]["accession"]
    details_resp = requests.get(f"{BASE_URL}/entry/interpro/{accession}")
    details = details_resp.json()
    name = details.get("metadata", {}).get("name", "N/A")
    entry_type = details.get("metadata", {}).get("type", "N/A")
    proteins_resp = requests.get(f"{BASE_URL}/protein/uniprot/entry/interpro/{accession}",
                                 params={"page_size": 1})
    proteins_count = proteins_resp.json().get("count", 0)
    print(f"{accession}\t{name}\t{entry_type}\t{proteins_count}")
