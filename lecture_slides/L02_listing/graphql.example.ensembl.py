import requests


query="""query getGene {
  gene(
    by_id: {genome_id: ensembl-genomeID,
            stable_id: "ENSG00000NNNNNN"}
  ) {
    alternative_symbols
    name
    so_term
    stable_id
    transcripts {
      stable_id
      symbol
    }
  }
}"""

url = "https://beta.ensembl.org/data/graphql"
response = requests.post(
               url,
               json={"query": query}
               )

if response.status_code == 200:
    data = response.json()
    gene = data['data']['gene']
    print("Gene name:", gene['name'])
    print("Stable ID:", gene['stable_id'])
    print("SO term:", gene['so_term'])
    print("Alternative symbols:", 
          gene['alternative_symbols'])
    print("Transcripts:")
    for t in gene['transcripts']:
        print(f"- {t['stable_id']} ({t['symbol']})")
