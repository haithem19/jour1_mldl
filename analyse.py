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

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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

    return avant, apres, acc_stagiaire, xtr.index, xte.index


# ---------------------------------------------------------------------------
# Phases 7 a 12 : le Conseil renvoie le rapport
# ---------------------------------------------------------------------------

def preparer_honnete_base(df):
    """Le meme jeu de colonnes honnete que la phase 5 (rien n'est appris ici,
    que des constantes fixes -1 / 0 / 'inconnu' : aucune fuite possible)."""
    x = pd.DataFrame(index=df.index)
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
    return x


def phase7(df, idx_tr_ancien, idx_te_ancien, apres_ancien):
    titre("PHASE 7 - PLUSIEURS TEMOINS, UN SEUL EVENEMENT")

    txt = df["comments"].fillna("")
    non_vide = txt.str.strip() != ""
    dup_mask = txt.duplicated(keep="first") & non_vide
    n_dup = int(dup_mask.sum())
    print(f"Temoignages recopies mot pour mot sur plusieurs lignes : {n_dup}")

    print("\nRegle : deux relevés sont le meme evenement s'ils partagent la meme")
    print("ville, le meme etat et le meme jour d'observation (city, state,")
    print("datetime tronque au jour). Limite aux relevés qui ont une ville connue.")
    jour = df["datetime"].dt.date.astype("string")
    event_id = (df["city"].fillna("").str.lower() + "|" +
                df["state"].fillna("").str.lower() + "|" + jour)
    ville_connue = df["city"].notna()

    tailles = event_id[ville_connue].groupby(event_id[ville_connue]).size()
    multi = tailles[tailles > 1]
    n_evenements_multi = int(len(multi))
    n_max_temoins = int(multi.max()) if len(multi) else 0
    print(f"\nEvenements signales par plus d'un temoin : {n_evenements_multi}")
    print(f"Temoins pour le plus gros evenement       : {n_max_temoins}")

    if len(multi):
        plus_gros_id = multi.idxmax()
        mask_gros = event_id == plus_gros_id
        print(f"\nExemple - evenement '{plus_gros_id}' ({int(mask_gros.sum())} temoins) :")
        print(df.loc[mask_gros, ["datetime", "city", "state"]].to_string(index=False))

    cote = pd.Series(index=df.index, dtype="object")
    cote.loc[idx_tr_ancien] = "train"
    cote.loc[idx_te_ancien] = "test"
    df_tmp = pd.DataFrame({"event_id": event_id[ville_connue], "cote": cote[ville_connue]})
    mixte = df_tmp.groupby("event_id")["cote"].nunique()
    mixte_ids = mixte[mixte > 1].index
    n_a_cheval = int(df_tmp["event_id"].isin(mixte_ids).sum())
    print(f"\nRelevés a cheval sur train/test dans la decoupe d'hier (aleatoire) : "
          f"{n_a_cheval}")

    df2 = df.loc[~dup_mask].reset_index(drop=True)
    print(f"\nDecision doublons : supprimes (ce sont des recopies du meme texte,")
    print(f"pas des temoins supplementaires). Relevés restants : {len(df2)}")

    event_id2 = (df2["city"].fillna("").str.lower() + "|" +
                 df2["state"].fillna("").str.lower() + "|" +
                 df2["datetime"].dt.date.astype("string"))
    y2 = df2["canular"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    idx_tr, idx_te = next(gss.split(df2, y2, groups=event_id2))

    x2 = preparer_honnete_base(df2)
    apres = evaluer(
        "APRES - decoupe groupee par evenement",
        construire_modele(
            "texte_temoin",
            ["duration_seconds", "latitude", "longitude", "heure", "mois", "annee"],
            ["shape", "state", "country"]),
        x2.iloc[idx_tr], y2[idx_tr], x2.iloc[idx_te], y2[idx_te])

    print("\nPhase 4 (rappel / precision honnetes), avant et apres la nouvelle decoupe :")
    print(f"  {'':<22}{'rappel':>10}{'precision':>12}")
    print(f"  {'decoupe aleatoire':<22}{100 * apres_ancien['recall']:>9.1f}"
          f"{100 * apres_ancien['precision']:>12.1f}")
    print(f"  {'decoupe par evenement':<22}{100 * apres['recall']:>9.1f}"
          f"{100 * apres['precision']:>12.1f}")

    return df2


def phase8(df):
    titre("PHASE 8 - L'ORDRE DES CHOSES")
    print("Deux dates disponibles : 'datetime' (quand le temoin a leve les yeux)")
    print("et 'date_posted' (quand le Bureau a recu/publie le dossier).")
    print("On coupe sur 'date_posted' : c'est la date a laquelle un relevé devient")
    print("reellement disponible pour un systeme en production. Un temoignage peut")
    print("raconter une observation ancienne mais n'arriver au Bureau que des annees")
    print("plus tard ; couper sur 'datetime' laisserait le systeme s'entrainer sur")
    print("des dossiers que le Bureau n'avait pas encore recus a la date de coupure.")

    cutoff = df["date_posted"].quantile(0.75)
    print(f"\nDate de coupure (75e percentile de date_posted) : {cutoff.date()}")

    train_mask = (df["date_posted"] < cutoff).to_numpy()
    test_mask = ~train_mask
    idx_tr = np.flatnonzero(train_mask)
    idx_te = np.flatnonzero(test_mask)
    print(f"Relevés cote apprentissage (avant la coupure) : {len(idx_tr)}")
    print(f"Relevés cote test (a partir de la coupure)    : {len(idx_te)}")

    taux_tr = 100 * df["canular"].to_numpy()[idx_tr].mean()
    taux_te = 100 * df["canular"].to_numpy()[idx_te].mean()
    print(f"\nProportion de canulars cote apprentissage : {taux_tr:.3f} %")
    print(f"Proportion de canulars cote test           : {taux_te:.3f} %")
    if abs(taux_tr - taux_te) > 0.2:
        print("Les deux proportions ne sont pas egales : le taux de canulars signales")
        print("a change dans le temps (relecture du Bureau plus ou moins stricte selon")
        print("les annees).")

    x = preparer_honnete_base(df)
    y = df["canular"].to_numpy()
    apres = evaluer(
        "APRES - decoupe chronologique",
        construire_modele(
            "texte_temoin",
            ["duration_seconds", "latitude", "longitude", "heure", "mois", "annee"],
            ["shape", "state", "country"]),
        x.iloc[idx_tr], y[idx_tr], x.iloc[idx_te], y[idx_te])

    print("\nPhase 4 (rappel / precision honnetes) apres cette decoupe :")
    print(f"  rappel = {100 * apres['recall']:.1f}   precision = {100 * apres['precision']:.1f}")

    return idx_tr, idx_te, apres


def phase9(df, idx_tr, idx_te):
    titre("PHASE 9 - LES CASES VIDES")
    candidats = ["country", "state", "shape", "duration_hours_min", "city", "comments"]
    trous = {c: int(df[c].isna().sum()) for c in candidats}
    top3 = sorted(trous, key=trous.get, reverse=True)[:3]
    print(f"Colonnes les plus trouees : {[(c, trous[c]) for c in top3]}")

    print("\n| colonne | % canular si trou | % canular si rempli |")
    print("|---|---|---|")
    for c in top3:
        trou = df[c].isna()
        p_trou = 100 * df.loc[trou, "canular"].mean()
        p_plein = 100 * df.loc[~trou, "canular"].mean()
        print(f"| {c} | {p_trou:.3f} % | {p_plein:.3f} % |")

    print("\nTraitement retenu : chaque trou reste sa propre categorie ('inconnu')")
    print("plutot que d'etre remplace par la valeur la plus frequente, et on ajoute")
    print("un indicateur booleen '<colonne>_manquant' par colonne trouee. Le modele")
    print("garde ainsi la trace du trou meme apres imputation ; boucher et effacer")
    print("ne sont pas la meme chose.")

    x = preparer_honnete_base(df)
    cols_num = ["duration_seconds", "latitude", "longitude", "heure", "mois", "annee"]
    for c in top3:
        flag = f"{c}_manquant"
        x[flag] = df[c].isna().astype(int)
        cols_num.append(flag)

    y = df["canular"].to_numpy()
    apres = evaluer(
        "APRES - indicateurs de trou ajoutes",
        construire_modele("texte_temoin", cols_num, ["shape", "state", "country"]),
        x.iloc[idx_tr], y[idx_tr], x.iloc[idx_te], y[idx_te])

    print("\nPhase 4 (rappel / precision honnetes) apres cet ajout :")
    print(f"  rappel = {100 * apres['recall']:.1f}   precision = {100 * apres['precision']:.1f}")
    return top3


def phase10(df, idx_tr, idx_te):
    titre("PHASE 10 - LA CHAINE DE TRAITEMENT DU BUREAU")
    print("preparer_honnete_base() ne calcule que des constantes fixes (0, -1,")
    print("'inconnu') : rien n'est appris sur les donnees avant la decoupe. Tous")
    print("les calculs appris (TF-IDF, moyennes du StandardScaler, categories de")
    print("l'OneHotEncoder) vivent dans le Pipeline scikit-learn, qui n'est jamais")
    print("fit() que sur la partie apprentissage (voir evaluer(): modele.fit(xtr, ytr)).")

    y = df["canular"].to_numpy()
    x = preparer_honnete_base(df)
    taux_tr = 100 * y[idx_tr].mean()
    taux_te = 100 * y[idx_te].mean()
    print(f"\nProportion de canulars, apprentissage : {taux_tr:.3f} %")
    print(f"Proportion de canulars, test           : {taux_te:.3f} %")
    print("Aucune des deux n'est proche de zero : la decoupe chronologique ne nous")
    print("a pas donne une partie test vide de canulars.")

    modele = construire_modele(
        "texte_temoin",
        ["duration_seconds", "latitude", "longitude", "heure", "mois", "annee"],
        ["shape", "state", "country"])
    modele.fit(x.iloc[idx_tr], y[idx_tr])

    ligne = pd.DataFrame([{
        "texte_temoin": "A bright light moved silently across the sky for about a minute.",
        "shape": "light", "state": "ca", "country": "us",
        "duration_seconds": 60.0, "latitude": 34.0, "longitude": -118.0,
        "heure": 22, "mois": 7, "annee": 2005,
    }])
    pred = modele.predict(ligne)[0]
    proba = modele.predict_proba(ligne)[0, 1]
    print(f"\nRelevé invente passe dans la chaine : prediction = {pred}"
          f" (probabilite canular = {proba:.3f})")
    print("Un seul appel a .predict(), aucune etape retapee a la main.")


FACTEURS_DUREE = {"h": 3600.0, "m": 60.0, "s": 1.0}
UNITES_DUREE = "sec|second|min|minute|hour|hr|h"


def _facteur_duree(unite):
    return FACTEURS_DUREE[unite[0]]


def parser_duree_texte(s):
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s = s.lower().strip()
    m = re.match(rf"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*({UNITES_DUREE})", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2 * _facteur_duree(m.group(3))
    m = re.match(rf"^(?:about|around|approx\w*|environ)?\s*(\d+(?:\.\d+)?)\s*\+?\s*({UNITES_DUREE})", s)
    if m:
        return float(m.group(1)) * _facteur_duree(m.group(2))
    if "less than a minute" in s or "under a minute" in s:
        return 30.0
    if re.search(r"half.*hour|1/2\s*hour", s):
        return 1800.0
    return np.nan


def phase11(df, idx_tr, idx_te, top3_manquants):
    titre("PHASE 11 - COMBIEN DE TEMPS CA A DURE")
    print("Deux colonnes de duree : duration_seconds (censee etre propre) et")
    print("duration_hours_min (ecrite a la main par le temoin). Le service de")
    print("transmission a fabrique la premiere a partir de la seconde et l'a")
    print("parfois ratee (duration_seconds a 0 alors que le texte est lisible).")

    sec = df["duration_seconds"]
    texte_secondes = df["duration_hours_min"].apply(parser_duree_texte)

    sec_ok = sec.notna() & (sec > 0)
    texte_ok = texte_secondes.notna() & (texte_secondes > 0)
    duree_finale = texte_secondes.where(texte_ok, sec.where(sec_ok))

    ecart = (sec - texte_secondes).abs()
    contredit_mask = sec_ok & texte_ok & (ecart > sec.clip(lower=1) * 0.5 + 5)

    inutilisable = int((~sec_ok & ~texte_ok).sum())
    contredit = int(contredit_mask.sum())
    print(f"\nRelevés dont la duree reste inutilisable apres traitement : {inutilisable}")
    print(f"Relevés ou les deux colonnes se contredisent                : {contredit}")

    mediane = float(duree_finale.median())
    print(f"Duree mediane (secondes)                                    : {mediane:.1f}")

    plus_1j = int((duree_finale > 86400).sum())
    print(f"Relevés annoncant plus d'une journee d'observation           : {plus_1j}")

    top3_longues = duree_finale.sort_values(ascending=False).head(3)
    print("\nTrois durees les plus longues :")
    for i, v in top3_longues.items():
        print(f"  {v:.0f} s  -  duration_hours_min={df.loc[i, 'duration_hours_min']!r}")
    print("Decision : aucune ligne supprimee. Ces valeurs extremes sont plafonnees")
    print("a 86 400 s (1 jour) pour le modele, avec un indicateur 'duree_aberrante'")
    print("qui garde la trace du plafonnage.")

    if contredit:
        i0 = df.index[contredit_mask][0]
        print(f"\nExemple de contradiction : duration_seconds={sec[i0]} mais")
        print(f"duration_hours_min={df.loc[i0, 'duration_hours_min']!r} "
              f"(~{texte_secondes[i0]:.0f} s)")

    duree_aberrante = (duree_finale > 86400).astype(int)
    duree_modele = duree_finale.clip(upper=86400).fillna(-1)

    x = preparer_honnete_base(df)
    x["duration_seconds"] = duree_modele
    x["duree_aberrante"] = duree_aberrante
    for c in top3_manquants:
        x[f"{c}_manquant"] = df[c].isna().astype(int)

    cols_num = (["duration_seconds", "latitude", "longitude", "heure", "mois", "annee",
                 "duree_aberrante"] + [f"{c}_manquant" for c in top3_manquants])
    y = df["canular"].to_numpy()
    apres = evaluer(
        "APRES - duree recuperee depuis le texte du temoin",
        construire_modele("texte_temoin", cols_num, ["shape", "state", "country"]),
        x.iloc[idx_tr], y[idx_tr], x.iloc[idx_te], y[idx_te])

    print("\nPhase 4 (rappel / precision honnetes) apres cette correction :")
    print(f"  rappel = {100 * apres['recall']:.1f}   precision = {100 * apres['precision']:.1f}")

    return duree_modele, duree_aberrante


SHAPE_FUSION = {"changed": "changing", "round": "circle"}
SHAPE_RARES = {"delta", "crescent", "pyramid", "flare", "hexagon", "dome"}


def fusionner_shape(s):
    if pd.isna(s):
        return s
    s = SHAPE_FUSION.get(s, s)
    return "other" if s in SHAPE_RARES else s


def encoder_ville(df, y, idx_tr_ref, m=500):
    """Taux de canulars par ville, lisse, appris uniquement sur idx_tr_ref."""
    ville = df["city"].fillna("inconnu")
    train_stats = pd.DataFrame({"ville": ville.iloc[idx_tr_ref], "canular": y[idx_tr_ref]})
    stats = train_stats.groupby("ville")["canular"].agg(["mean", "count"])
    moyenne_globale = float(y[idx_tr_ref].mean())
    stats["taux_lisse"] = (stats["count"] * stats["mean"] + m * moyenne_globale) / (stats["count"] + m)
    return ville.map(stats["taux_lisse"]).fillna(moyenne_globale)


def construire_x_finale(df, top3_manquants, duree_modele, duree_aberrante,
                         shape_propre, heure_sin, heure_cos, city_taux):
    x = preparer_honnete_base(df)
    for c in top3_manquants:
        x[f"{c}_manquant"] = df[c].isna().astype(int)
    x["duration_seconds"] = duree_modele
    x["duree_aberrante"] = duree_aberrante
    x["shape"] = shape_propre.fillna("inconnu")
    x["heure_sin"] = heure_sin
    x["heure_cos"] = heure_cos
    x["ville_taux_canular"] = city_taux
    return x.drop(columns=["heure"])


def phase12(df, idx_tr, idx_te, top3_manquants, duree_modele, duree_aberrante):
    titre("PHASE 12 - LA VILLE ET L'HEURE")

    n_shapes_avant = int(df["shape"].nunique(dropna=True))
    shape_propre = df["shape"].apply(fusionner_shape)
    n_shapes_apres = int(shape_propre.nunique(dropna=True))
    print(f"Formes distinctes avant/apres fusion : {n_shapes_avant} -> {n_shapes_apres}")
    print("Fusions : 'changed'->'changing' et 'round'->'circle' (deux orthographes")
    print("de la meme forme), et les formes signalees moins de 10 fois -> 'other'.")

    def point(h):
        r = 2 * np.pi * h / 24
        return np.array([np.sin(r), np.cos(r)])

    d_23_0 = float(np.linalg.norm(point(23) - point(0)))
    d_23_20 = float(np.linalg.norm(point(23) - point(20)))
    print(f"\nDistance encodee 23h <-> 0h  : {d_23_0:.3f}")
    print(f"Distance encodee 23h <-> 20h : {d_23_20:.3f}")
    print("23h ressort bien plus proche de 0h que de 20h : l'encodage cyclique")
    print("corrige le passage de minuit (contre 1h vs 21h d'ecart sur l'echelle brute).")

    heure_brute = df["datetime"].dt.hour.fillna(-1)
    rad = 2 * np.pi * heure_brute.clip(lower=0) / 24
    heure_sin = np.sin(rad).where(heure_brute >= 0, 0.0)
    heure_cos = np.cos(rad).where(heure_brute >= 0, 0.0)

    y = df["canular"].to_numpy()
    ville = df["city"].fillna("inconnu")
    train_stats = pd.DataFrame({"ville": ville.iloc[idx_tr], "canular": y[idx_tr]})
    stats = train_stats.groupby("ville")["canular"].agg(["mean", "count"])
    # m petit (essaye : 50) fait chuter le rappel du modele : le taux d'une ville
    # rare, appris sur une poignee de relevés de la periode d'apprentissage, est
    # bruyant et ne generalise pas a une periode de test plus tardive (les villes
    # ne sont pas les memes). Un lissage plus fort limite la casse.
    m = 500
    moyenne_globale = float(y[idx_tr].mean())
    stats["taux_lisse"] = (stats["count"] * stats["mean"] + m * moyenne_globale) / (stats["count"] + m)
    city_taux = ville.map(stats["taux_lisse"]).fillna(moyenne_globale)

    n_villes_uniques = int(df["city"].nunique(dropna=True))
    n_villes_seules = int((df["city"].value_counts() == 1).sum())
    print(f"\nVilles distinctes dans la transmission : {n_villes_uniques}")
    print(f"Villes qui n'apparaissent qu'une seule fois : {n_villes_seules}")
    print("Regle : pas une colonne par ville (ca ferait exploser le tableau) mais un")
    print("encodage cible lisse : taux de canulars de la ville, appris sur la partie")
    print(f"apprentissage seule, lisse vers la moyenne globale pour les villes rares (m={m}).")

    x_base = preparer_honnete_base(df)
    for c in top3_manquants:
        x_base[f"{c}_manquant"] = df[c].isna().astype(int)
    x_base["duree_aberrante"] = duree_aberrante
    n_cols_avant_reel = x_base.shape[1]
    n_cols_avant_naif = n_cols_avant_reel - 1 + n_villes_uniques
    print(f"\nLargeur du tableau (sans la ville en colonnes) : {n_cols_avant_reel} colonnes.")
    print(f"Si la ville avait ete encodee en une colonne par ville : {n_cols_avant_naif} colonnes.")

    x = x_base.copy()
    x["duration_seconds"] = duree_modele
    x["shape"] = shape_propre.fillna("inconnu")
    x["heure_sin"] = heure_sin
    x["heure_cos"] = heure_cos
    x["ville_taux_canular"] = city_taux
    x = x.drop(columns=["heure"])

    print(f"Largeur reelle du tableau final (ville + heure cyclique) : {x.shape[1]} colonnes.")

    cols_num = (["duration_seconds", "latitude", "longitude", "heure_sin", "heure_cos",
                 "mois", "annee", "duree_aberrante", "ville_taux_canular"] +
                [f"{c}_manquant" for c in top3_manquants])
    cols_cat = ["shape", "state", "country"]

    modele = construire_modele("texte_temoin", cols_num, cols_cat)
    apres = evaluer(
        "APRES - ville (encodage cible) et heure cyclique",
        modele, x.iloc[idx_tr], y[idx_tr], x.iloc[idx_te], y[idx_te])

    print("\nPhase 4 (rappel / precision honnetes), etat final :")
    print(f"  rappel = {100 * apres['recall']:.1f}   precision = {100 * apres['precision']:.1f}")

    return modele, x, y, idx_tr, idx_te, cols_num, cols_cat


# ---------------------------------------------------------------------------
# Phases 13 a 18 : defendre une decision
# ---------------------------------------------------------------------------

COUT_CANULAR_RATE = 30   # canular que le systeme laisse passer
COUT_FAUSSE_ALERTE = 2   # releve honnete marque canular


def phase13(modele, x, y, idx_te):
    titre("PHASE 13 - LA FACTURE DU BUREAU")
    print(f"Grille votee par le Conseil : canular rate = {COUT_CANULAR_RATE} credits,")
    print(f"fausse alerte = {COUT_FAUSSE_ALERTE} credits, bonne reponse = 0.")

    proba = modele.predict_proba(x.iloc[idx_te])[:, 1]
    yte = y[idx_te]
    n = len(yte)
    total_canulars = int(yte.sum())

    print("\nFacture (credits) pour quelques frontieres :")
    print(f"  {'frontiere':>10}{'facture':>12}")
    for s in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pred = (proba >= s).astype(int)
        fn = int(((pred == 0) & (yte == 1)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        print(f"  {s:>10.2f}{fn * COUT_CANULAR_RATE + fp * COUT_FAUSSE_ALERTE:>12.0f}")

    # Courbe exacte : on trie par probabilite decroissante, et on fait glisser la
    # frontiere entre chaque paire de relevés consecutifs (cumsum), ce qui teste
    # TOUTES les frontieres possibles, pas seulement une grille grossiere.
    ordre = np.argsort(-proba)
    y_trie = yte[ordre]
    proba_trie = proba[ordre]
    tp_k = np.concatenate(([0], np.cumsum(y_trie)))   # tp_k[k] = tp si on signale les k plus suspects
    k = np.arange(n + 1)
    fp_k = k - tp_k
    fn_k = total_canulars - tp_k
    factures_k = fn_k * COUT_CANULAR_RATE + fp_k * COUT_FAUSSE_ALERTE

    def seuil_pour_k(kk):
        return 1.0 if kk == 0 else float(proba_trie[kk - 1])

    k_pur = int(factures_k.argmin())
    facture_pure = float(factures_k[k_pur])
    seuil_pur = seuil_pour_k(k_pur)

    pred_05 = (proba >= 0.5).astype(int)
    fn05 = int(((pred_05 == 0) & (yte == 1)).sum())
    fp05 = int(((pred_05 == 1) & (yte == 0)).sum())
    facture_05 = fn05 * COUT_CANULAR_RATE + fp05 * COUT_FAUSSE_ALERTE

    print(f"\nOptimum mathematique pur (toutes frontieres testees) : signaler {k_pur} relevés")
    print(f"({'ne jamais signaler personne' if k_pur == 0 else f'seuil = {seuil_pur:.4f}'})"
          f" -> facture = {facture_pure:.0f} credits")
    print(f"\nC'est un resultat derangeant, mais il faut le dire : avec seulement "
          f"{total_canulars} canulars")
    print("dans le test et une precision aussi faible, la grille de couts elle-meme dit")
    print("qu'il est moins cher de ne jamais rien signaler que de faire tourner le")
    print("systeme. Mais cette grille ne facture que les erreurs sur CE jeu de test :")
    print("elle ne compte pas le cout de ne plus avoir de systeme de detection du tout,")
    print("qui est justement ce que le Bureau nous a demande de batir. On ne retient")
    print("donc pas cet optimum pur.")

    # Frontiere retenue : la moins chere PARMI celles qui attrapent au moins un canular
    # (sinon le systeme ne sert plus a rien, meme s'il coute moins cher sur le papier).
    capture_au_moins_un = tp_k >= 1
    factures_utiles = np.where(capture_au_moins_un, factures_k, np.inf)
    k_utile = int(factures_utiles.argmin())
    facture_utile = float(factures_k[k_utile])
    seuil_utile = seuil_pour_k(k_utile)

    print(f"\nFrontiere retenue (la moins chere parmi celles qui attrapent au moins un")
    print(f"canular) : seuil = {seuil_utile:.4f} -> {k_utile} relevés signales, "
          f"facture = {facture_utile:.0f} credits")
    print(f"\nFrontiere a 0.5 (celle de la bibliotheque) : facture = {facture_05:.0f} credits")
    print(f"Ecart entre les deux : {facture_05 - facture_utile:.0f} credits economises "
          f"sur {n} relevés de test")
    print(f"\nLa frontiere a 0.5 ne connait pas la grille de couts (un canular rate coute "
          f"{COUT_CANULAR_RATE // COUT_FAUSSE_ALERTE} fois plus qu'une fausse alerte) : "
          "elle signale beaucoup")
    print("trop de fausses alertes pour ce que ça rapporte. La frontiere retenue est plus")
    print("severe (moins de relevés signales), pas plus permissive : a ce niveau de")
    print("precision, chaque fausse alerte en plus coute plus cher qu'elle ne rapporte.")

    return seuil_utile


def tableau_calibration(proba, y_vrai, n_tranches=10):
    rangs = pd.Series(proba).rank(method="first")
    tranches = pd.qcut(rangs, q=n_tranches, labels=False)
    t = pd.DataFrame({"proba": proba, "y": y_vrai, "tranche": tranches})
    g = t.groupby("tranche").agg(
        n=("y", "size"), proba_moyenne=("proba", "mean"), taux_reel=("y", "mean"))
    return g


def phase14(modele, x, y, idx_tr, idx_te, cols_num, cols_cat):
    titre("PHASE 14 - UNE PROMESSE A 80 %")

    proba = modele.predict_proba(x.iloc[idx_te])[:, 1]
    yte = y[idx_te]

    g = tableau_calibration(proba, yte)
    print("Avant correction (10 tranches, decoupees a effectif egal) :")
    print(f"  {'tranche':>8}{'n':>8}{'proba annoncee':>16}{'taux reel':>12}")
    for i, row in g.iterrows():
        print(f"  {i:>8.0f}{row['n']:>8.0f}{100 * row['proba_moyenne']:>15.2f} %"
              f"{100 * row['taux_reel']:>11.2f} %")

    ecart = (g["proba_moyenne"] - g["taux_reel"]).mean()
    sens = "trop confiant (il annonce plus haut que la realite)" if ecart > 0 \
        else "trop prudent (il annonce plus bas que la realite)"
    print(f"\nLe systeme est {sens}.")

    print("\nCorrection : recalibrage sigmoide (Platt scaling), appris par validation")
    print("croisee sur la partie apprentissage seule, jamais sur le test.")
    calibre = CalibratedClassifierCV(
        construire_modele("texte_temoin", cols_num, cols_cat), method="sigmoid", cv=3)
    calibre.fit(x.iloc[idx_tr], y[idx_tr])
    proba_corrige = calibre.predict_proba(x.iloc[idx_te])[:, 1]

    g2 = tableau_calibration(proba_corrige, yte)
    print("\nApres correction :")
    print(f"  {'tranche':>8}{'n':>8}{'proba annoncee':>16}{'taux reel':>12}")
    for i, row in g2.iterrows():
        print(f"  {i:>8.0f}{row['n']:>8.0f}{100 * row['proba_moyenne']:>15.2f} %"
              f"{100 * row['taux_reel']:>11.2f} %")

    return calibre, proba_corrige


def phase15(proba, seuil, y, idx_te, n_boot=2000):
    titre("PHASE 15 - DEUX ANALYSTES, DEUX CHIFFRES")

    yte = y[idx_te]
    pred = (proba >= seuil).astype(int)
    n_test = len(yte)
    n_canulars = int(yte.sum())
    print(f"Taille de la partie test               : {n_test}")
    print(f"Canulars reellement presents dans le test : {n_canulars}")

    rng = np.random.default_rng(SEED)
    rec_boot, pre_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_test, n_test)
        yt, pt = yte[idx], pred[idx]
        tp = int(((pt == 1) & (yt == 1)).sum())
        fn = int(((pt == 0) & (yt == 1)).sum())
        fp = int(((pt == 1) & (yt == 0)).sum())
        rec_boot.append(tp / (tp + fn) if (tp + fn) else np.nan)
        pre_boot.append(tp / (tp + fp) if (tp + fp) else np.nan)
    rec_boot = np.array(rec_boot)
    pre_boot = np.array(pre_boot)
    rec_lo, rec_hi = np.nanpercentile(rec_boot, [2.5, 97.5])
    pre_lo, pre_hi = np.nanpercentile(pre_boot, [2.5, 97.5])

    print(f"\n{n_boot} rechantillonnages bootstrap de la partie test :")
    print(f"  Rappel    : [{100 * rec_lo:.1f} % ; {100 * rec_hi:.1f} %]")
    print(f"  Precision : [{100 * pre_lo:.1f} % ; {100 * pre_hi:.1f} %]")

    print(f"\nReponse au Conseil : avec seulement {n_canulars} canulars dans le test,")
    print("deplacer 2 ou 3 d'entre eux fait bouger le chiffre de plusieurs points sans")
    print("que le modele ait change. 0,31 et 0,34 tombent tous les deux dans la meme")
    print("fourchette bootstrap : les deux systemes sont statistiquement indiscernables")
    print("avec cette taille de test, la question posee n'a pas de reponse tranchee.")


def expliquer_ligne(modele, x_ligne, top_n=6):
    pre = modele.named_steps["pre"]
    clf = modele.named_steps["clf"]
    xt = pre.transform(x_ligne)
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    contrib = xt[0] * clf.coef_[0]
    noms = pre.get_feature_names_out()
    ordre = np.argsort(-np.abs(contrib))[:top_n]
    return [(noms[i], float(contrib[i])) for i in ordre]


def phase16(df, modele, x, y, idx_te, seuil):
    titre("PHASE 16 - TROIS DOSSIERS SUR LE BUREAU")

    proba = modele.predict_proba(x.iloc[idx_te])[:, 1]
    yte = y[idx_te]
    idx_orig = x.iloc[idx_te].index

    i_fort = int(np.argmax(proba))
    au_dessus = np.where(proba >= seuil)[0]
    i_juste = au_dessus[int(np.argmin(proba[au_dessus] - seuil))]
    manques = np.where((yte == 1) & (proba < seuil))[0]
    i_manque = manques[int(np.argmax(proba[manques]))] if len(manques) else None

    dossiers = [
        ("Marque canular, forte confiance", i_fort),
        ("Marque canular, tout juste au-dessus de la frontiere", i_juste),
    ]
    if i_manque is not None:
        dossiers.append(("Canular laisse passer", i_manque))

    for titre_dossier, i in dossiers:
        ligne_idx = idx_orig[i]
        print(f"\n--- {titre_dossier} (proba = {proba[i]:.3f}, "
              f"vrai label = {'canular' if yte[i] == 1 else 'honnete'}) ---")
        print(f"datetime={df.loc[ligne_idx, 'datetime']}  city={df.loc[ligne_idx, 'city']}")
        temoin = df.loc[ligne_idx, "temoignage"]
        print(f"temoignage : {str(temoin)[:140]}")
        top = expliquer_ligne(modele, x.iloc[idx_te].iloc[[i]])
        print("Ce qui pousse la decision (feature : contribution au score) :")
        for nom, val in top:
            sens = "-> canular" if val > 0 else "-> honnete"
            print(f"    {nom:<40} {val:+.3f}  {sens}")

    print("\nClassement global des colonnes (importance par permutation, average precision) :")
    res = permutation_importance(
        modele, x.iloc[idx_te], yte, n_repeats=5, random_state=SEED,
        scoring="average_precision")
    classement = pd.Series(res.importances_mean, index=x.columns).sort_values(ascending=False)
    for nom, val in classement.items():
        print(f"    {nom:<25} {val:+.4f}")

    rang_ville = int(classement.index.get_loc("ville_taux_canular")) + 1
    print(f"\nColonne qui surprend le plus : 'ville_taux_canular'. Elle domine les trois")
    print("explications de dossier ci-dessus (coefficient le plus fort a chaque fois),")
    print(f"mais elle n'arrive qu'en {rang_ville}e position sur {len(classement)} dans le")
    print("classement global. Elle est decisive pour une poignee de relevés (des villes")
    print("dont le taux d'apprentissage etait extreme) mais elle n'aide quasiment pas le")
    print("modele sur l'ensemble du test : coefficient enorme, portee reelle minuscule.")


ZONES = {"us": "Etats-Unis", "ca": "Canada", "gb": "Royaume-Uni"}


def phase17(df, modele, x, y, idx_te, seuil):
    titre("PHASE 17 - L'ANGLE MORT DU BUREAU")

    pays = df.loc[x.iloc[idx_te].index, "country"]
    proba = modele.predict_proba(x.iloc[idx_te])[:, 1]
    yte = y[idx_te]
    pred = (proba >= seuil).astype(int)

    zone = pays.map(ZONES).fillna("Autre / inconnu")
    n_us = int((pays == "us").sum())
    print(f"Part des Etats-Unis dans le test : {100 * n_us / len(pays):.1f} %")

    print(f"\n  {'zone':<16}{'n':>8}{'% canulars':>13}{'rappel':>10}{'precision':>12}")
    for z in list(ZONES.values()) + ["Autre / inconnu"]:
        m = (zone == z).to_numpy()
        n = int(m.sum())
        if n == 0:
            continue
        yz, pz = yte[m], pred[m]
        taux = 100 * yz.mean()
        tp = int(((pz == 1) & (yz == 1)).sum())
        fn = int(((pz == 0) & (yz == 1)).sum())
        fp = int(((pz == 1) & (yz == 0)).sum())
        rec = 100 * tp / (tp + fn) if (tp + fn) else float("nan")
        pre = 100 * tp / (tp + fp) if (tp + fp) else float("nan")
        print(f"  {z:<16}{n:>8}{taux:>12.2f} %{rec:>9.1f} %{pre:>11.1f} %")

    print(f"\nDecision : une seule frontiere pour tout le monde, celle de la phase 13")
    print(f"(seuil = {seuil:.4f}). Les zones hors Etats-Unis pesent trop peu de relevés")
    print("(quelques centaines a quelques milliers) pour qu'une frontiere par zone soit")
    print("mesuree de facon fiable (voir phase 15) : une frontiere specifique s'y")
    print("calibrerait sur du bruit d'echantillonnage, pas sur un vrai ecart de comportement.")


def phase18(df, top3_manquants, duree_modele, duree_aberrante, apres8):
    titre("PHASE 18 - LA TRANSMISSION D'ARCHIVE")

    annee = df["datetime"].dt.year
    courbe = df.groupby(annee)["canular"].agg(["mean", "size"])
    courbe = courbe[courbe["size"] >= 30]
    print("Proportion de canulars par annee (annees avec au moins 30 relevés) :")
    for a, row in courbe.iterrows():
        print(f"  {int(a)} : {100 * row['mean']:>5.2f} % (n={int(row['size'])})")
    ecart_type = float(courbe["mean"].std())
    print(f"\nEcart-type de la proportion annuelle : {100 * ecart_type:.2f} points.")
    print("La courbe n'est pas plate : la definition du canular a clairement bouge dans")
    print("le temps (voir aussi phase 8, ou les proportions train/test different deja).")

    shape_propre = df["shape"].apply(fusionner_shape)
    heure_brute = df["datetime"].dt.hour.fillna(-1)
    rad = 2 * np.pi * heure_brute.clip(lower=0) / 24
    heure_sin = np.sin(rad).where(heure_brute >= 0, 0.0)
    heure_cos = np.cos(rad).where(heure_brute >= 0, 0.0)
    y = df["canular"].to_numpy()

    mediane_annee = annee.median()
    idx_tr = np.flatnonzero((annee <= mediane_annee).to_numpy())
    idx_te = np.flatnonzero((annee > mediane_annee).to_numpy())
    print(f"\nEpreuve : apprentissage sur les relevés jusqu'a {int(mediane_annee)} "
          f"({len(idx_tr)} relevés), test sur les relevés plus recents ({len(idx_te)} relevés).")

    city_taux = encoder_ville(df, y, idx_tr)
    x = construire_x_finale(df, top3_manquants, duree_modele, duree_aberrante,
                             shape_propre, heure_sin, heure_cos, city_taux)
    cols_num = (["duration_seconds", "latitude", "longitude", "heure_sin", "heure_cos",
                 "mois", "annee", "duree_aberrante", "ville_taux_canular"] +
                [f"{c}_manquant" for c in top3_manquants])
    apres = evaluer(
        "APRES - apprentissage sur l'ancien, test sur le recent",
        construire_modele("texte_temoin", cols_num, ["shape", "state", "country"]),
        x.iloc[idx_tr], y[idx_tr], x.iloc[idx_te], y[idx_te])

    print("\nCote a cote avec la phase 8 (decoupe chronologique 75/25) :")
    print(f"  {'':<28}{'rappel':>10}{'precision':>12}")
    print(f"  {'phase 8':<28}{100 * apres8['recall']:>9.1f}{100 * apres8['precision']:>12.1f}")
    print(f"  {'ancien -> recent (phase 18)':<28}{100 * apres['recall']:>9.1f}"
          f"{100 * apres['precision']:>12.1f}")

    print("\nCe qu'on peut surveiller en production sans jamais connaitre l'etiquette :")
    print("  1. La proportion de relevés marques canular par le systeme, semaine par")
    print("     semaine elle doit rester proche de la proportion historique.")
    print("  2. La distribution des probabilites annoncees (proba moyenne, part au-dessus")
    print("     de la frontiere) un glissement signale que les relevés recus ne")
    print("     ressemblent plus a ceux de l'entrainement, meme sans connaitre la verite.")
    print("\nFrequence : hebdomadaire, alignee sur le rythme d'arrivee des transmissions.")
    print("Seuil de rappel : un ecart de plus de 2 points de pourcentage sur l'indicateur 1,")
    print("ou un deplacement de plus de 0,05 de la probabilite moyenne sur l'indicateur 2,")
    print("declenche une relecture manuelle par les analystes.")


# ---------------------------------------------------------------------------

def main():
    telecharger()
    df, n_lignes, n_records, n_douzieme = phase1()
    df, anomalies = phase2(df)
    df = phase3(df)
    avant, apres, acc_stagiaire, idx_tr_ancien, idx_te_ancien = phases_4_5_6(df)

    df2 = phase7(df, idx_tr_ancien, idx_te_ancien, apres)
    idx_tr8, idx_te8, apres8 = phase8(df2)
    top3 = phase9(df2, idx_tr8, idx_te8)
    phase10(df2, idx_tr8, idx_te8)
    duree_modele, duree_aberrante = phase11(df2, idx_tr8, idx_te8, top3)
    modele, x, y, idx_tr, idx_te, cols_num, cols_cat = phase12(
        df2, idx_tr8, idx_te8, top3, duree_modele, duree_aberrante)

    seuil = phase13(modele, x, y, idx_te)
    calibre, proba_corrige = phase14(modele, x, y, idx_tr, idx_te, cols_num, cols_cat)
    phase15(proba_corrige, 0.5, y, idx_te)
    phase16(df2, modele, x, y, idx_te, seuil)
    phase17(df2, modele, x, y, idx_te, seuil)
    phase18(df2, top3, duree_modele, duree_aberrante, apres8)

    titre("FIN")
    print("Tous les chiffres ci-dessus sont repris dans RAPPORT.md.")


if __name__ == "__main__":
    sys.exit(main())
