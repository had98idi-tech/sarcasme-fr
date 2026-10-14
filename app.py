"""Petite démo Gradio : on tape un titre, le modèle dit s'il est satirique.
lancer avec : python app.py
"""
import gradio as gr
from transformers import pipeline

MODELE = "models/camembert-satire"  # ou le nom du modèle sur Hugging Face
clf = pipeline("text-classification", model=MODELE, tokenizer=MODELE, top_k=None)


def predire(titre):
    # on renvoie un dict {label: proba} pour l'affichage gradio
    return {r["label"]: r["score"] for r in clf(titre)[0]}


demo = gr.Interface(
    fn=predire,
    inputs=gr.Textbox(label="Titre d'article", placeholder="Ex : Le Sénat vote la suppression des lundis"),
    outputs=gr.Label(label="Prédiction"),
    title="Détecteur de titres satiriques (français)",
    description="CamemBERT fine-tuné sur FreSaDa. Attention : marche moins bien sur des titres récents ou d'autres sources (voir le README).",
    examples=["Le Sénat vote la suppression des lundis",
              "Le gouvernement présente son budget 2025",
              "Un homme découvre qu'il travaille depuis 10 ans dans une entreprise qui n'existe pas"],
)

if __name__ == "__main__":
    demo.launch()
