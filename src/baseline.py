"""Baseline : TF-IDF + régression logistique."""
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

# mots + bigrammes, on garde les majuscules en minuscule
vect = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vect.fit_transform(train["texte"])

clf = LogisticRegression(max_iter=1000, C=5)
clf.fit(X_train, train["label"])

scores = {}
for nom, df in [("val", val), ("test", test)]:
    pred = clf.predict(vect.transform(df["texte"]))
    scores[nom] = {"accuracy": round(accuracy_score(df["label"], pred), 4),
                   "f1_macro": round(f1_score(df["label"], pred, average="macro"), 4)}
    df.assign(pred=pred).to_csv(f"results/preds_baseline_{nom}.csv", index=False)
print(scores)

# les mots qui comptent le plus pour chaque classe (intéressant pour l'analyse)
mots = vect.get_feature_names_out()
coefs = clf.coef_[0]
ordre = coefs.argsort()
scores["top_satirique"] = [mots[i] for i in ordre[::-1][:25]]
scores["top_normal"] = [mots[i] for i in ordre[:25]]
print("satirique :", scores["top_satirique"])
print("normal :", scores["top_normal"])

with open("results/scores_baseline.json", "w") as f:
    json.dump(scores, f, ensure_ascii=False, indent=2)
