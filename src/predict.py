"""Tester le modèle sur une phrase.

usage : python src/predict.py "Le gouvernement interdit officiellement les lundis"
"""
import sys
from transformers import pipeline

MODELE = "models/camembert-satire"

clf = pipeline("text-classification", model=MODELE, tokenizer=MODELE)

phrases = sys.argv[1:] or [
    "Le gouvernement interdit officiellement les lundis",
    "La réforme des retraites adoptée au Sénat",
]
for p in phrases:
    res = clf(p)[0]
    print(f"{res['label']:>10} ({res['score']:.2f})  |  {p}")
