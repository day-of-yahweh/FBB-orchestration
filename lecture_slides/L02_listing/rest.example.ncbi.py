import requests

taxon_name = "Escherichia coli"
params = {"db": "assembly", "term": f'"{taxon_name}"[Organism]',
          "retmax": 20, "retmode": "json"}
url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
response = requests.get(url_search, params=params)
data = response.json()

assembly_ids = data["esearchresult"].get("idlist", [])
print("Assembly IDs:", assembly_ids)
params_summary = {"db": "assembly", 
                  "id": ",".join(assembly_ids),
                  "retmode": "json"}
url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
response_summary = requests.get(url_summary, params=params_summary)
summary = response_summary.json()["result"]
