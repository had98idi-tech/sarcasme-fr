"""Préparation des données FreSaDa (titres d'articles satiriques vs normaux)."""
import pandas as pd
from sklearn.model_selection import train_test_split

DOSSIER = "data/FreSaDa/data"


def charger(split):
    # quoting=3 -> on ignore les guillemets sinon pandas coupe certains titres
    df = pd.read_csv(f"{DOSSIER}/{split}/summary.tsv", sep="\t", quoting=3)
    df = df.rename(columns={"title": "texte"})
    # dans FreSaDa : 1 = satirique, -1 = normal. on passe en 1 / 0
    df["label"] = (df["label"] == 1).astype(int)
    # on enlève les guillemets en trop autour de certains titres
    df["texte"] = df["texte"].str.strip('"').str.replace('""', '"').str.strip()
    # il y a beaucoup de doublons (ex : un titre répété 148 fois dans le test) -> on les enlève
    avant = len(df)
    df = df.drop_duplicates("texte")
    print(f"{split} : {avant} -> {len(df)} titres après suppression des doublons")
    return df[["texte", "label"]]


if __name__ == "__main__":
    train = charger("train")
    test = charger("test")

    # validation = 10% du train (même sources que le train)
    train, val = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=42)

    # le test de FreSaDa vient d'autres sites que le train -> c'est un test "cross-domain"
    for nom, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(f"data/{nom}.csv", index=False)
        print(nom, len(df), "| % satirique :", round(df["label"].mean() * 100, 1))
