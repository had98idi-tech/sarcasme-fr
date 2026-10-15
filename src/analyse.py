"""Analyse linguistique : quels indices distinguent les titres satiriques,
et est-ce qu'ils restent les mêmes entre le train et le test ?
+ analyse des erreurs de CamemBERT sur le test.
"""
import json, re
import pandas as pd
import matplotlib.pyplot as plt

BLEU, ORANGE, GRIS = "#2a78d6", "#eb6834", "#6b6a63"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# indices linguistiques repérés avec des regex simples
INDICES = {
    "question (?)": r"\?",
    "exclamation (!)": r"!",
    "deux-points (:)": r":",
    "guillemets / citation": r"[«»\"“”]",
    "chiffre": r"\d",
    "1re/2e personne (je, nous, vous…)": r"\b(?:je|j'|j’|nous|vous|mon|ma|mes|notre|votre)\b",
    "comment / pourquoi / voici": r"\b(?:comment|pourquoi|voici)\b",
    "verbe de révélation (avoue, découvre…)": r"\b(?:avoue|découvre|annonce|révèle|confirme|apprend)\b",
    "il / elle + verbe (récit)": r"\b(?:il|elle)\b",
}


def taux_indices(df):
    # % de titres qui contiennent chaque indice, par classe
    res = {}
    for nom, motif in INDICES.items():
        a = df["texte"].str.contains(motif, flags=re.I, regex=True)
        res[nom] = {"normal": round(a[df.label == 0].mean() * 100, 1),
                    "satirique": round(a[df.label == 1].mean() * 100, 1)}
    return res


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
analyse = {"indices_train": taux_indices(train), "indices_test": taux_indices(test)}

for nom, df in [("train", train), ("test", test)]:
    n = df["texte"].str.split().str.len()
    analyse[f"longueur_moyenne_{nom}"] = {"normal": round(n[df.label == 0].mean(), 1),
                                          "satirique": round(n[df.label == 1].mean(), 1)}

# --- figure 1 : indices par classe, train vs test ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
noms = list(INDICES)
y = range(len(noms))
for ax, split in zip(axes, ["train", "test"]):
    d = analyse[f"indices_{split}"]
    ax.barh([i + 0.2 for i in y], [d[k]["normal"] for k in noms], 0.38, color=BLEU, label="normal")
    ax.barh([i - 0.2 for i in y], [d[k]["satirique"] for k in noms], 0.38, color=ORANGE, label="satirique")
    ax.set_title(f"{split.capitalize()} ({'mêmes sources' if split == 'train' else 'autres sources'})")
    ax.set_xlabel("% des titres contenant l'indice")
    ax.grid(axis="x", color="#e5e4df"); ax.set_axisbelow(True)
axes[0].set_yticks(list(y)); axes[0].set_yticklabels(noms); axes[0].invert_yaxis()
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Les indices linguistiques ne sont pas les mêmes selon les sources", fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/indices_linguistiques.png", dpi=150)

# --- figure 2 : comparaison des scores ---
sb = json.load(open("results/scores_baseline.json"))
sc = json.load(open("results/scores_camembert.json"))
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for j, (nom, s, c) in enumerate([("TF-IDF + rég. log.", sb, BLEU), ("CamemBERT", sc, ORANGE)]):
    vals = [s["val"]["f1_macro"] * 100, s["test"]["f1_macro"] * 100]
    xs = [i + (j - 0.5) * 0.36 for i in range(2)]
    ax.bar(xs, vals, 0.34, color=c, label=nom)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1, f"{v:.0f}", ha="center", color="#2b2a27")
ax.axhline(50, color=GRIS, ls="--", lw=1); ax.text(1.62, 51, "hasard", color=GRIS, ha="right")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Validation\n(mêmes sources)", "Test\n(autres sources)"])
ax.set_ylabel("F1 macro (%)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper right"); ax.set_title("Scores : forte chute hors domaine", fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/scores.png", dpi=150)

# --- analyse des erreurs de CamemBERT sur le test ---
p = pd.read_csv("results/preds_camembert_test.csv")
p["erreur"] = p.pred != p.label
# lignes = vrai label, colonnes = prédiction
analyse["matrice_confusion_test"] = {"vrai_normal": [int(((p.label == 0) & (p.pred == k)).sum()) for k in (0, 1)],
                                    "vrai_satirique": [int(((p.label == 1) & (p.pred == k)).sum()) for k in (0, 1)]}
# taux d'erreur selon la présence de chaque indice
analyse["taux_erreur_par_indice"] = {}
for nom, motif in INDICES.items():
    a = p["texte"].str.contains(motif, flags=re.I, regex=True)
    if a.sum() >= 30:
        analyse["taux_erreur_par_indice"][nom] = {"avec": round(p[a].erreur.mean() * 100, 1),
                                                 "sans": round(p[~a].erreur.mean() * 100, 1),
                                                 "n_avec": int(a.sum())}
# les erreurs les plus "sûres d'elles" = les plus intéressantes à lire
fp = p[(p.label == 0) & (p.pred == 1)].sort_values("proba", ascending=False)
fn = p[(p.label == 1) & (p.pred == 0)].sort_values("proba")
analyse["exemples_faux_positifs"] = fp.head(12)[["texte", "proba"]].values.tolist()
analyse["exemples_faux_negatifs"] = fn.head(12)[["texte", "proba"]].values.tolist()

json.dump(analyse, open("results/analyse.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(analyse, ensure_ascii=False, indent=1))
