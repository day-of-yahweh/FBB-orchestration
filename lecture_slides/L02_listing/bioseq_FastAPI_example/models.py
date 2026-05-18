from pydantic import BaseModel, Field

class SequenceInput(BaseModel):
    sequence: str = Field(..., example="ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
    alphabet: str = Field("dna", example="dna")

class GCResponse(BaseModel):
    gc_content: float

class TranslationResponse(BaseModel):
    protein: str

class ORFResponse(BaseModel):
    start: int
    end: int
    protein: str
