from Bio.Seq import Seq

def gc_content(seq: str) -> float:
    seq = seq.upper()
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq) * 100 if seq else 0.0


def translate_dna(seq: str) -> str:
    return str(Seq(seq).translate(to_stop=True))


def find_orfs(seq: str, min_len: int = 30):
    seq = Seq(seq)
    orfs = []
    for frame in range(3):
        protein = seq[frame:].translate()
        aa_start = 0
        for part in protein.split("*"):
            if len(part) >= min_len:
                start = frame + aa_start * 3
                end = start + len(part) * 3
                orfs.append((start, end, str(part)))
            aa_start += len(part) + 1
    return orfs
