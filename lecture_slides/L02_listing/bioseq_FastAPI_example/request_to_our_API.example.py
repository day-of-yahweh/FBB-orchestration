import requests

url = "http://127.0.0.1:8000/gc"

payload = {
    "sequence": "ATGGCCATTGTAATGGGCCGC",
    "alphabet": "dna"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
