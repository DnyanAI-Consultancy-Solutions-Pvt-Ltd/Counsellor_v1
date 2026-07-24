from pathlib import Path

def classify_document(filename: str, sample: str) -> str:
    s=(filename+" "+sample[:4000]).lower()
    rules=[("seat_matrix",["seat matrix","intake capacity","sanctioned intake"]),("cutoff",["cut off","cutoff","merit score","cap round"]),("cap_brochure",["information brochure","admission rules","eligibility","centralized admission process"]),("institute_list",["institute code","college code","list of institutes"])]
    for kind,words in rules:
        if any(w in s for w in words): return kind
    return "mhtcet_reference"
