"""Fine-tuning de CamemBERT pour détecter les titres satiriques.

usage : python src/train_camembert.py [modele_de_base]
(par défaut : camembert-base depuis Hugging Face)
"""
import sys, json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

BASE = sys.argv[1] if len(sys.argv) > 1 else "camembert-base"
SORTIE = "models/camembert-satire"
torch.manual_seed(42)

tok = AutoTokenizer.from_pretrained(BASE)
# on donne des noms clairs aux labels (sinon on a LABEL_0 / LABEL_1)
modele = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2,
    id2label={0: "normal", 1: "satirique"},
    label2id={"normal": 0, "satirique": 1},
    ignore_mismatched_sizes=True)


def prep(nom):
    df = pd.read_csv(f"data/{nom}.csv")
    ds = Dataset.from_pandas(df.rename(columns={"label": "labels"}))
    # les titres sont courts, 64 tokens c'est largement suffisant
    return ds.map(lambda b: tok(b["texte"], truncation=True, max_length=64), batched=True)


train, val, test = prep("train"), prep("val"), prep("test")


def metriques(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, pred),
            "f1_macro": f1_score(labels, pred, average="macro")}


args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=2,
    learning_rate=2e-5,          # valeur classique pour BERT
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=70,             # ~10% des steps au début avec un lr qui monte doucement
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to=[],
)

trainer = Trainer(model=modele, args=args, train_dataset=train, eval_dataset=val,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metriques)
trainer.train()

# évaluation finale + sauvegarde des prédictions pour l'analyse d'erreurs
scores = {}
for nom, ds in [("val", val), ("test", test)]:
    out = trainer.predict(ds)
    probas = torch.softmax(torch.tensor(out.predictions), -1)[:, 1].numpy()
    scores[nom] = {k.replace("test_", ""): round(v, 4) for k, v in out.metrics.items()
                   if k in ("test_accuracy", "test_f1_macro")}
    pd.read_csv(f"data/{nom}.csv").assign(pred=(probas > 0.5).astype(int), proba=probas.round(4)) \
        .to_csv(f"results/preds_camembert_{nom}.csv", index=False)
print(scores)
json.dump(scores, open("results/scores_camembert.json", "w"), indent=2)

trainer.save_model(SORTIE)
tok.save_pretrained(SORTIE)
print("modèle sauvegardé dans", SORTIE)
