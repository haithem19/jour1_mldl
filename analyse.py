"""
Bureau d'Analyse Terrestre (OVNI) - Partie 1
Reception des releves de la sonde Klaxo-3.

Ce script tourne d'une traite dans un dossier vide :
il telecharge la transmission, la charge, la type, fabrique une etiquette
"canular", entraine deux modeles (avec puis sans fuite temporelle) et
compare le tout au modele du stagiaire.

    python analyse.py

Dependances : pandas, scikit-learn
"""

import csv
import io
import os
import re
import sys
import urllib.request

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = (
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/"
    "csv-data/ufo-complete-geocoded-time-standardized.csv"
)
FICHIER = "releves_klaxo3.csv"

# Le service de transmission a oublie les en-tetes. Manifeste retrouve a part.
COLONNES = [
    "datetime", "city", "state", "country", "shape", "duration_seconds",
    "duration_hours_min", "comments", "date_posted", "latitude", "longitude",
]

SEED = 42


def titre(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Recuperer la transmission
# ---------------------------------------------------------------------------

def telecharger():
    if os.path.exists(FICHIER):
        print(f"{FICHIER} deja present ({os.path.getsize(FICHIER)} octets), "
              "pas de retelechargement.")
        return
    print(f"Telechargement de la transmission depuis {URL} ...")
    urllib.request.urlretrieve(URL, FICHIER)
    print(f"Recu : {os.path.getsize(FICHIER)} octets.")


# ---------------------------------------------------------------------------
# Phase 1 : ouvrir la caisse
# ---------------------------------------------------------------------------

def phase1():
    titre("PHASE 1 - OUVRIR LA CAISSE")

    with open(FICHIER, "rb") as f:
        brut = f.read()

    # Nombre de lignes physiques du fichier (le fichier finit par un saut de ligne).
    lignes_physiques = brut.count(b"\n") + (0 if brut.endswith(b"\n") else 1)

    # pandas.read_csv() casse sur ce fichier : certaines lignes ont 12 champs.
    # On passe donc par le module csv, qui lit chaque enregistrement tel quel
    # (les guillemets protegent les virgules et les sauts de ligne du temoignage).
    texte = brut.decode("utf-8")
    enregistrements = list(csv.reader(io.StringIO(texte)))

    normales = [r for r in enregistrements if len(r) == 11]
    douzieme = [r for r in enregistrements if len(r) == 12]
    autres = [r for r in enregistrements if len(r) not in (11, 12)]

    print(f"Lignes physiques dans le fichier          : {lignes_physiques}")
    print(f"Enregistrements CSV lus                   : {len(enregistrements)}")
    print(f"  - a 11 champs (conformes au manifeste)  : {len(normales)}")
    print(f"  - a 12 champs (traitees a part)         : {len(douzieme)}")
    print(f"  - autres nombres de champs              : {len(autres)}")

    if douzieme:
        print("\nUne ligne problematique, telle qu'elle arrive :")
        exemple = douzieme[2] if len(douzieme) > 2 else douzieme[0]
        print("  " + repr(exemple))
        print("\nCe qui cloche : 12 champs au lieu de 11. La signature est toujours")
        print("la meme sur ces lignes : city vide, country vide, un champ vide")
        print("supplementaire en position 5, latitude=0 et longitude=0.")
        print("La forme de l'objet et la duree ecrite se retrouvent decalees d'un cran.")

    # Reparation : on remet ces 196 lignes sur 11 champs. On jette le champ vide
    # parasite et on va rechercher shape / duration a leur position reelle.
    # (indices : 0 datetime, 1 city, 2 state, 3 country, 4 vide, 5 duration_seconds,
    #            6 shape, 7 duration_hours_min, 8 comments, 9 date_posted, 10 lat, 11 lon)
    reparees = []
    for r in enregistrements:
        if len(r) == 12:
            r = [r[0], r[1], r[2], r[3], r[6], r[5], r[7], r[8], r[9], r[10], r[11]]
        reparees.append(r)

    df = pd.DataFrame(reparees, columns=COLONNES)
    print(f"\nCharges en memoire au final               : {len(df)} releves")
    print("Aucune ligne n'a ete supprimee : les 196 lignes mises de cote ont ete")
    print("recollees sur le bon schema, pas jetees.")
    return df, lignes_physiques, len(enregistrements), len(douzieme)


# ---------------------------------------------------------------------------
# Phase 2 : rien n'est du bon type
# ---------------------------------------------------------------------------

def phase2(df):
    titre("PHASE 2 - RIEN N'EST DU BON TYPE")

    anomalies = {}

    # --- Champs numeriques -------------------------------------------------
    for col in ["latitude", "longitude", "duration_seconds"]:
        brut = df[col]
        num = pd.to_numeric(brut, errors="coerce")
        resiste = num.isna() & brut.notna()
        n = int(resiste.sum())
        anomalies[col] = n
        print(f"\n[{col}] valeurs qui resistent a la conversion : {n}")
        if n:
            vc = brut[resiste].value_counts()
            for val, cnt in vc.items():
                print(f"    {val!r:>18}  x{cnt}")
        df[col] = num

    # --- Champs dates ------------------------------------------------------
    # datetime au format m/j/AAAA HH:MM, date_posted au format m/j/AAAA.
    dt = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    resiste_dt = dt.isna()
    n_dt = int(resiste_dt.sum())
    anomalies["datetime"] = n_dt
    print(f"\n[datetime] valeurs qui resistent a la conversion : {n_dt}")
    print("    exemples :", df.loc[resiste_dt, "datetime"].head(5).tolist())
    n_24 = int(df["datetime"].str.contains(r"24:00$", na=False).sum())
    print(f"    dont heure ecrite '24:00' (heure qui n'existe pas) : {n_24}")

    # Choix : on repare 24:00 en 00:00 du lendemain plutot que de perdre 1262 dates.
    a_reparer = resiste_dt & df["datetime"].str.contains(r"24:00$", na=False)
    repare = pd.to_datetime(
        df.loc[a_reparer, "datetime"].str.replace(" 24:00", " 00:00", regex=False),
        format="%m/%d/%Y %H:%M", errors="coerce",
    ) + pd.Timedelta(days=1)
    dt.loc[a_reparer] = repare
    print(f"    reparees (24:00 -> 00:00 du lendemain) : {int(a_reparer.sum())}")
    print(f"    restent NaT : {int(dt.isna().sum())}")
    df["datetime"] = dt

    dp = pd.to_datetime(df["date_posted"], format="%m/%d/%Y", errors="coerce")
    n_dp = int(dp.isna().sum())
    anomalies["date_posted"] = n_dp
    print(f"\n[date_posted] valeurs qui resistent a la conversion : {n_dp}")
    df["date_posted"] = dp

    # --- Autres anomalies de nature differente -----------------------------
    print("\nAutres anomalies reperees (pas des echecs de conversion) :")

    n_zero = int(((df["latitude"] == 0) & (df["longitude"] == 0)).sum())
    print(f"  - coordonnees exactement (0, 0) [golfe de Guinee] : {n_zero}")
    anomalies["latlon_zero"] = n_zero

    n_ent = int(df["comments"].str.contains(r"&#\d+|&amp;|&quot;", na=False).sum())
    print(f"  - temoignages avec entites HTML non decodees (&#44, &amp;) : {n_ent}")
    anomalies["entites_html"] = n_ent

    vides = {c: int((df[c] == "").sum()) for c in
             ["city", "state", "country", "shape", "duration_hours_min", "comments"]}
    print(f"  - champs texte vides : {vides}")
    anomalies["vides"] = vides

    # On decode les entites et on normalise les vides en NaN.
    df["comments"] = (df["comments"].astype("string")
                      .str.replace("&#44", ",", regex=False)
                      .str.replace("&#39", "'", regex=False)
                      .str.replace("&#33", "!", regex=False)
                      .str.replace("&amp;", "&", regex=False)
                      .str.replace("&quot;", '"', regex=False))
    for c in ["city", "state", "country", "shape", "duration_hours_min", "comments"]:
        df[c] = df[c].astype("string").replace("", pd.NA)

    print("\nTypes finaux :")
    print(df.dtypes.to_string())
    return df, anomalies


# ---------------------------------------------------------------------------
# Phase 3 : fabriquer l'etiquette canular
# ---------------------------------------------------------------------------

NOTE = re.compile(r"\(\(.*?\)\)", re.S)


def phase3(df):
    titre("PHASE 3 - FABRIQUER L'ETIQUETTE CANULAR")

    txt = df["comments"].fillna("")
    # Le temoignage contient parfois un commentaire editorial entre doubles
    # parentheses, ajoute apres coup par un employe du Bureau.
    df["note_bureau"] = txt.apply(lambda s: " ".join(NOTE.findall(s)))
    df["temoignage"] = txt.apply(lambda s: NOTE.sub(" ", s).strip())

    df["canular"] = df["note_bureau"].str.lower().str.contains("hoax").astype(int)

    n = int(df["canular"].sum())
    print("Regle : un releve est marque canular si le commentaire editorial entre")
    print("        doubles parentheses ajoute par le Bureau contient le mot 'hoax'.")
    print(f"\nReleves marques canular : {n}")
    print(f"Proportion              : {100 * n / len(df):.3f} %")
    print(f"Releves portant une note editoriale (quelle qu'elle soit) : "
          f"{int((df['note_bureau'].str.len() > 0).sum())}")

    rate = int(df["temoignage"].str.lower().str.contains("hoax").sum())
    print(f"\nLimite : {rate} releves ecrivent 'hoax' dans le temoignage lui-meme")
    print("sans note du Bureau -> la regle les rate. Et surtout, la regle ne")
    print("detecte pas un canular : elle recopie l'avis d'un employe qui a lu le")
    print("dossier. Un canular jamais relu par personne est etiquete 'pas canular'.")

    print("\nExemples marques canular :")
    for s in df.loc[df["canular"] == 1, "comments"].head(3):
        print("  *", s[:110])
    return df


# ---------------------------------------------------------------------------
# Phases 4 et 5 : le modele, avec puis sans fuite
# ---------------------------------------------------------------------------

def preparer_features(df):
    x = pd.DataFrame(index=df.index)
    x["texte_complet"] = df["comments"].fillna("")
    x["texte_temoin"] = df["temoignage"].fillna("")
    x["shape"] = df["shape"].fillna("inconnu")
    x["state"] = df["state"].fillna("inconnu")
    x["country"] = df["country"].fillna("inconnu")
    x["duration_seconds"] = df["duration_seconds"].fillna(-1)
    x["latitude"] = df["latitude"].fillna(0)
    x["longitude"] = df["longitude"].fillna(0)
    x["heure"] = df["datetime"].dt.hour.fillna(-1)
    x["mois"] = df["datetime"].dt.month.fillna(-1)
    x["annee"] = df["datetime"].dt.year.fillna(-1)
    # Delai de traitement : ecrit par l'employe, pas disponible a l'instant t.
    x["delai_jours"] = (df["date_posted"] - df["datetime"]).dt.days.fillna(-1)
    x["annee_posted"] = df["date_posted"].dt.year.fillna(-1)
    x["a_note_bureau"] = (df["note_bureau"].str.len() > 0).astype(int)
    return x


def construire_modele(col_texte, cols_num, cols_cat):
    pre = ColumnTransformer([
        ("txt", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                min_df=3, max_features=60000,
                                sublinear_tf=True), col_texte),
        ("num", StandardScaler(), cols_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cols_cat),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced", solver="liblinear")),
    ])


def evaluer(nom, modele, xtr, ytr, xte, yte):
    modele.fit(xtr, ytr)
    pred = modele.predict(xte)
    rec = recall_score(yte, pred, zero_division=0)
    pre = precision_score(yte, pred, zero_division=0)
    acc = accuracy_score(yte, pred)
    print(f"\n[{nom}]")
    print(f"  Sur 100 canulars reellement presents, attrapes : {100 * rec:.1f}")
    print(f"  Sur 100 releves signales, vrais canulars       : {100 * pre:.1f}")
    print(f"  Taux de bonnes reponses (accuracy)             : {100 * acc:.2f} %")
    return {"recall": rec, "precision": pre, "accuracy": acc, "pred": pred}


def phases_4_5_6(df):
    x = preparer_features(df)
    y = df["canular"].to_numpy()

    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y)

    titre("PHASE 4 - LE PREMIER VERDICT (toutes les colonnes)")
    print(f"Apprentissage sur {len(xtr)} releves, evaluation sur {len(xte)} releves")
    print("jamais vus pendant l'apprentissage (decoupage stratifie 75/25, seed 42).")
    print(f"Canulars dans le jeu de test : {int(yte.sum())}")

    avant = evaluer(
        "AVANT - avec ce que l'employe du Bureau a ecrit",
        construire_modele(
            "texte_complet",
            ["duration_seconds", "latitude", "longitude", "heure", "mois",
             "annee", "delai_jours", "annee_posted", "a_note_bureau"],
            ["shape", "state", "country"]),
        xtr, ytr, xte, yte)

    titre("PHASE 5 - LE CONSEIL NE VOUS CROIT PAS")
    print("Colonnes retirees : la note editoriale du Bureau (extraite du champ")
    print("comments), le drapeau 'a_note_bureau', date_posted et le delai de")
    print("traitement. Il reste ce qui existe le soir meme de l'observation.")

    apres = evaluer(
        "APRES - seulement ce qui existe a l'instant du signalement",
        construire_modele(
            "texte_temoin",
            ["duration_seconds", "latitude", "longitude", "heure", "mois", "annee"],
            ["shape", "state", "country"]),
        xtr, ytr, xte, yte)

    print("\nCote a cote :")
    print(f"  {'':<12}{'rappel':>10}{'precision':>12}")
    print(f"  {'avant':<12}{100 * avant['recall']:>9.1f}{100 * avant['precision']:>12.1f}")
    print(f"  {'apres':<12}{100 * apres['recall']:>9.1f}{100 * apres['precision']:>12.1f}")

    titre("PHASE 6 - LE MODELE LE PLUS BETE DU BUREAU")
    pred_stagiaire = [0] * len(yte)
    acc_stagiaire = accuracy_score(yte, pred_stagiaire)
    print("Systeme du stagiaire : repondre toujours 'ce n'est pas un canular'.")
    print(f"  Taux de bonnes reponses du stagiaire : {100 * acc_stagiaire:.2f} %")
    print(f"  Taux de bonnes reponses du modele    : {100 * apres['accuracy']:.2f} %")
    print(f"  Rappel du stagiaire                  : "
          f"{100 * recall_score(yte, pred_stagiaire, zero_division=0):.1f}")
    print(f"  Rappel du modele                     : {100 * apres['recall']:.1f}")
    print("\nLe stagiaire a un tres bon taux de bonnes reponses parce que 99 % des")
    print("releves ne sont pas des canulars. Il n'en attrape aucun. La mesure a")
    print("presenter au Conseil est le rappel sur la classe canular (et la precision")
    print("qui va avec), pas le taux de bonnes reponses.")

    return avant, apres, acc_stagiaire


# ---------------------------------------------------------------------------

def main():
    telecharger()
    df, n_lignes, n_records, n_douzieme = phase1()
    df, anomalies = phase2(df)
    df = phase3(df)
    phases_4_5_6(df)
    titre("FIN")
    print("Tous les chiffres ci-dessus sont repris dans RAPPORT.md.")


if __name__ == "__main__":
    sys.exit(main())
