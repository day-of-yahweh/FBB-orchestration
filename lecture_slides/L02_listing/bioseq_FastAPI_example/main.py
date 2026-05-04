from fastapi import FastAPI, HTTPException
from models import SequenceInput, GCResponse, TranslationResponse, ORFResponse
import bio

app = FastAPI(
    title="BioSeq API",
    description="REST API for basic bioinformatics sequence analysis",
    version="1.0"
)


@app.post("/gc", response_model=GCResponse)
def compute_gc(data: SequenceInput):
    if data.alphabet.lower() != "dna":
        raise HTTPException(status_code=400, detail="GC content only for DNA")
    gc = bio.gc_content(data.sequence)
    return GCResponse(gc_content=gc)


@app.post("/translate", response_model=TranslationResponse)
def translate(data: SequenceInput):
    if data.alphabet.lower() != "dna":
        raise HTTPException(status_code=400, detail="Translation only for DNA")
    protein = bio.translate_dna(data.sequence)
    return TranslationResponse(protein=protein)


@app.post("/orfs", response_model=list[ORFResponse])
def find_orfs(data: SequenceInput, min_len: int = 30):
    if data.alphabet.lower() != "dna":
        raise HTTPException(status_code=400, detail="ORF search only for DNA")
    orfs = bio.find_orfs(data.sequence, min_len)
    return [
        ORFResponse(start=s, end=e, protein=p)
        for s, e, p in orfs
    ]
