# Rapport au Conseil — Réception des relevés Klaxo-3

Tous les chiffres ci-dessous sont produits par `analyse.py`, qui se relance
d'une traite dans un dossier vide (téléchargement compris).

---

## Phase 1 — Ouvrir la caisse

| | |
|---|---|
| Lignes dans le fichier | **88 875** |
| Relevés chargés en mémoire | **88 875** |
| Lignes traitées à part | **196** |

Personne n'y arrivait parce que `pandas.read_csv()` refuse le fichier : 196 lignes
ont 12 champs au lieu de 11. J'ai donc lu le fichier avec le module `csv`, ligne par
ligne, sans imposer de largeur. Comme ça rien ne disparaît en silence.

Une ligne problématique, telle qu'elle arrive :

```
['10/14/2011 22:30', '', 'nv', '', '', '0', 'light', '22', '3 Green lights', '10/19/2011', '0', '0']
```

Ce qui cloche : il y a un champ vide en trop, en 5ᵉ position. Du coup tout ce qui
suit est décalé d'un cran — la forme (`light`) tombe dans la case durée, la durée
(`22`) tombe dans la case témoignage, etc. Les 196 lignes ont exactement la même
signature : ville vide, pays vide, le champ vide en trop, `duration_seconds` à 0,
et latitude/longitude à 0. C'est visiblement le lot des relevés que le géocodeur
n'a pas su placer.

**Ce que j'en ai fait :** je ne les ai pas jetées. Je les remets sur 11 champs en
supprimant le champ parasite et en allant rechercher `shape` et
`duration_seconds` à leur vraie position. Total final : 88 875 relevés, soit
88 679 + 196. Les trois nombres tombent juste.

---

## Phase 2 — Rien n'est du bon type

Aucune ligne supprimée à cette étape.

| Champ | Valeurs qui résistent | Ce qu'il y a dedans |
|---|---|---|
| `latitude` | **1** | `33q.200088` |
| `longitude` | 0 | — |
| `duration_seconds` | **5** | `2\``, `8\``, `0.5\``, et 2 valeurs vides |
| `datetime` | **1 262** | toutes finissent par `24:00` |
| `date_posted` | 0 | — |

### Les anomalies, par nature et par coupable

**1. `33q.200088` dans la latitude — 1 valeur sur 88 875.**
Un `q` s'est glissé au milieu d'un nombre. C'est cette *seule* valeur qui rendait
toute la colonne latitude inutilisable : pandas la lit en texte, et donc plus aucune
carte n'est traçable. Origine : **service de transmission** (corruption d'un
caractère, ce n'est ni une faute d'orthographe humaine ni une mesure).

**2. L'heure `24:00` — 1 262 relevés.**
`24:00` n'existe pas dans un format horaire. Les témoins écrivent ça pour dire
« minuit ». Origine : **témoin**.
*Décision :* je répare (`24:00` → `00:00` du lendemain) plutôt que de perdre
1 262 dates. Après réparation, 0 date manquante.

**3. Des accents graves dans la durée — 3 valeurs (`2\``, `8\``, `0.5\``).**
Quelqu'un a tapé une apostrophe (pour « minutes ») qui est ressortie en backtick,
collée au nombre. Origine : **témoin** (avec un coup de main du **service de
transmission** pour le caractère). 2 durées sont simplement vides.

**4. Coordonnées exactement (0, 0) — 1 690 relevés.**
Le point (0, 0) est en plein golfe de Guinée. Aucun de ces relevés n'y a eu lieu :
c'est la valeur par défaut quand le géocodage échoue. Origine : **capteur**.
Sur une carte, ça fait 1 690 OVNIs dans l'océan.

**5. Entités HTML non décodées dans les témoignages — 37 035 relevés.**
On lit `&#44` au lieu d'une virgule, `&#39` au lieu d'une apostrophe, `&amp;` au
lieu de `&`. Origine : **service de transmission** (passage par une page web mal
désencodée). Je les décode.

**6. Champs vides :** `city` 196, `state` 7 519, `country` 12 561, `shape` 3 006,
`duration_hours_min` 3 108, `comments` 35. Origine mélangée : le témoin ne remplit
pas tout, le géocodeur ne trouve pas toujours le pays.

Types finaux : `datetime` et `date_posted` en datetime, `latitude`, `longitude`,
`duration_seconds` en float, le reste en string.

---

## Phase 3 — Trier les canulars

**La règle, en une phrase :** un relevé est un canular si le commentaire éditorial
entre doubles parenthèses ajouté au témoignage par un employé du Bureau contient le
mot « hoax ».

Concrètement ça ressemble à `((HOAX??))` en tête de témoignage, ou à
`((NUFORC Note: Possible hoax?? PD))` à la fin.

| | |
|---|---|
| Relevés marqués canular | **785** |
| Proportion | **0,883 %** |
| Relevés portant une note éditoriale quelconque | 3 745 |

**Ce que la règle rate ou attrape à tort :**

- 30 relevés écrivent « hoax » dans le témoignage lui-même sans note du Bureau
  (par exemple un titre « Lights in Irvine October 2007: Hoax »). Ma règle les rate.
- La règle attrape aussi les `((Possible hoax??))` — un doute, pas une certitude.
  Donc une partie des 785 sont peut-être authentiques.
- Le vrai problème est plus grave : **cette étiquette ne détecte pas les canulars,
  elle recopie l'avis d'un employé qui a lu le dossier.** Un canular que personne
  n'a jamais relu est étiqueté « pas un canular ». Le 0,883 % est donc un plancher,
  pas le taux réel de canulars.

---

## Phase 4 — Le premier verdict

Modèle : régression logistique sur TF-IDF du témoignage (1-2 grammes) + forme,
état, pays, durée, coordonnées, heure/mois/année, délai de traitement.

Découpage **75 / 25 stratifié, seed 42**. Le modèle apprend sur **66 656** relevés
et est évalué sur **22 219 relevés qu'il n'a jamais vus**, dont **196 canulars**.
Tous les chiffres ci-dessous viennent de ce jeu de test.

| | |
|---|---|
| Sur 100 canulars réellement présents, attrapés | **100,0** |
| Sur 100 relevés signalés, vrais canulars | **96,1** |
| Taux de bonnes réponses | 99,96 % |

C'est trop beau. Voir phase 5.

---

## Phase 5 — Le Conseil ne nous croit pas

La conseillère a raison. Voici qui écrit quoi.

| La colonne | Qui écrit cette information | À quel moment | Cette personne savait-elle déjà s'il s'agissait d'un canular ? |
|---|---|---|---|
| `comments` — partie témoignage | le témoin | le soir de l'observation | non |
| `comments` — note `((...))` du Bureau | un employé du Bureau | des semaines plus tard, après lecture du dossier | **OUI** |
| `a_note_bureau` (présence d'une note) | un employé du Bureau | des semaines plus tard | **OUI** |
| `date_posted` | le Bureau, à la publication | des semaines plus tard | **OUI** (il a traité le dossier) |
| `delai_jours` (date_posted − datetime) | dérivé de `date_posted` | des semaines plus tard | **OUI** |
| `datetime` (heure, mois, année) | le témoin | le soir de l'observation | non |
| `shape` | le témoin | le soir de l'observation | non |
| `duration_seconds` | le témoin | le soir de l'observation | non |
| `city`, `state` | le témoin | le soir de l'observation | non |
| `country` | le géocodeur / capteur | à la réception | non |
| `latitude`, `longitude` | le géocodeur / capteur | à la réception | non |

Sortent du modèle : la note `((...))`, le drapeau de présence de note, `date_posted`
et le délai. Je ne supprime pas `comments` en entier — je le coupe en deux et je
garde uniquement ce que le témoin a écrit.

### Avant / après, côte à côte

| | Sur 100 canulars, attrapés (rappel) | Sur 100 signalés, vrais (précision) |
|---|---|---|
| **Avant** (avec ce que l'employé a écrit) | 100,0 | 96,1 |
| **Après** (seulement ce qui existe le soir même) | **8,2** | **4,0** |

### L'explication

Le premier chiffre n'avait pas le droit d'exister parce que l'étiquette « canular »
et la variable la plus utile du modèle sont **le même texte** : le mot « hoax » écrit
par l'employé du Bureau. Le modèle ne prédisait rien, il relisait la réponse dans sa
propre entrée — un employé avait déjà tranché avant lui, des semaines plus tôt.
Les 8,2 et 4,0 sont le vrai niveau du problème : à partir du seul témoignage brut,
distinguer un canular d'un signalement sincère est presque impossible, parce que les
deux sont écrits de la même façon par des gens qui racontent la même histoire.

---

## Phase 6 — Le modèle le plus bête du Bureau

Système du stagiaire : répondre « ce n'est pas un canular », toujours.

| Système | Taux de bonnes réponses | Rappel sur les canulars |
|---|---|---|
| **Stagiaire** (toujours « non ») | **99,12 %** | 0,0 |
| Notre modèle honnête (phase 5) | 97,46 % | 8,2 |

Sur le taux de bonnes réponses, le stagiaire nous bat. Et pourtant son score ne
prouve strictement rien : 99,12 % des relevés du jeu de test ne sont pas des
canulars, donc dire « non » à tout le monde donne mécaniquement 99,12 %. Il n'a
attrapé **aucun** des 196 canulars. Son système est exactement aussi utile qu'une
pierre, et une pierre a 99,12 % de bonnes réponses.

**La mesure que je présente au Conseil : le rappel sur la classe canular, accompagné
de sa précision.** C'est la seule qui répond à la question posée — « combien de
canulars attrapez-vous » — et la seule que le stagiaire ne peut pas gagner : son
rappel est zéro par construction. Le taux de bonnes réponses ne mesure ici que le
déséquilibre du fichier, pas la qualité d'un système ; sur une classe à 0,9 %, il est
saturé avant même qu'on ait commencé à travailler.

**Ce que ça veut dire honnêtement :** avec 8,2 de rappel et 4,0 de précision, notre
modèle n'est pas encore utilisable en production. Mais il est *vrai*, et le 100/96
de la phase 4 ne l'était pas.
