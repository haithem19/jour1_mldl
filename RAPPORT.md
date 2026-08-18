# Rapport au Conseil Réception des relevés Klaxo-3

Tous les chiffres ci-dessous sont produits par `analyse.py`, qui se relance
d'une traite dans un dossier vide (téléchargement compris).

---

## Phase 1 Ouvrir la caisse

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
suit est décalé d'un cran la forme (`light`) tombe dans la case durée, la durée
(`22`) tombe dans la case témoignage, etc. Les 196 lignes ont exactement la même
signature : ville vide, pays vide, le champ vide en trop, `duration_seconds` à 0,
et latitude/longitude à 0. C'est visiblement le lot des relevés que le géocodeur
n'a pas su placer.

**Ce que j'en ai fait :** je ne les ai pas jetées. Je les remets sur 11 champs en
supprimant le champ parasite et en allant rechercher `shape` et
`duration_seconds` à leur vraie position. Total final : 88 875 relevés, soit
88 679 + 196. Les trois nombres tombent juste.

---

## Phase 2 Rien n'est du bon type

Aucune ligne supprimée à cette étape.

| Champ | Valeurs qui résistent | Ce qu'il y a dedans |
|---|---|---|
| `latitude` | **1** | `33q.200088` |
| `longitude` | 0 | |
| `duration_seconds` | **5** | `2\``, `8\``, `0.5\``, et 2 valeurs vides |
| `datetime` | **1 262** | toutes finissent par `24:00` |
| `date_posted` | 0 | |

### Les anomalies, par nature et par coupable

**1. `33q.200088` dans la latitude 1 valeur sur 88 875.**
Un `q` s'est glissé au milieu d'un nombre. C'est cette *seule* valeur qui rendait
toute la colonne latitude inutilisable : pandas la lit en texte, et donc plus aucune
carte n'est traçable. Origine : **service de transmission** (corruption d'un
caractère, ce n'est ni une faute d'orthographe humaine ni une mesure).

**2. L'heure `24:00` 1 262 relevés.**
`24:00` n'existe pas dans un format horaire. Les témoins écrivent ça pour dire
« minuit ». Origine : **témoin**.
*Décision :* je répare (`24:00` → `00:00` du lendemain) plutôt que de perdre
1 262 dates. Après réparation, 0 date manquante.

**3. Des accents graves dans la durée 3 valeurs (`2\``, `8\``, `0.5\``).**
Quelqu'un a tapé une apostrophe (pour « minutes ») qui est ressortie en backtick,
collée au nombre. Origine : **témoin** (avec un coup de main du **service de
transmission** pour le caractère). 2 durées sont simplement vides.

**4. Coordonnées exactement (0, 0) 1 690 relevés.**
Le point (0, 0) est en plein golfe de Guinée. Aucun de ces relevés n'y a eu lieu :
c'est la valeur par défaut quand le géocodage échoue. Origine : **capteur**.
Sur une carte, ça fait 1 690 OVNIs dans l'océan.

**5. Entités HTML non décodées dans les témoignages 37 035 relevés.**
On lit `&#44` au lieu d'une virgule, `&#39` au lieu d'une apostrophe, `&amp;` au
lieu de `&`. Origine : **service de transmission** (passage par une page web mal
désencodée). Je les décode.

**6. Champs vides :** `city` 196, `state` 7 519, `country` 12 561, `shape` 3 006,
`duration_hours_min` 3 108, `comments` 35. Origine mélangée : le témoin ne remplit
pas tout, le géocodeur ne trouve pas toujours le pays.

Types finaux : `datetime` et `date_posted` en datetime, `latitude`, `longitude`,
`duration_seconds` en float, le reste en string.

---

## Phase 3 Trier les canulars

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
- La règle attrape aussi les `((Possible hoax??))` un doute, pas une certitude.
  Donc une partie des 785 sont peut-être authentiques.
- Le vrai problème est plus grave : **cette étiquette ne détecte pas les canulars,
  elle recopie l'avis d'un employé qui a lu le dossier.** Un canular que personne
  n'a jamais relu est étiqueté « pas un canular ». Le 0,883 % est donc un plancher,
  pas le taux réel de canulars.

---

## Phase 4 Le premier verdict

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

## Phase 5 Le Conseil ne nous croit pas

La conseillère a raison. Voici qui écrit quoi.

| La colonne | Qui écrit cette information | À quel moment | Cette personne savait-elle déjà s'il s'agissait d'un canular ? |
|---|---|---|---|
| `comments` partie témoignage | le témoin | le soir de l'observation | non |
| `comments` note `((...))` du Bureau | un employé du Bureau | des semaines plus tard, après lecture du dossier | **OUI** |
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
et le délai. Je ne supprime pas `comments` en entier je le coupe en deux et je
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
propre entrée un employé avait déjà tranché avant lui, des semaines plus tôt.
Les 8,2 et 4,0 sont le vrai niveau du problème : à partir du seul témoignage brut,
distinguer un canular d'un signalement sincère est presque impossible, parce que les
deux sont écrits de la même façon par des gens qui racontent la même histoire.

---

## Phase 6 Le modèle le plus bête du Bureau

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
de sa précision.** C'est la seule qui répond à la question posée « combien de
canulars attrapez-vous » et la seule que le stagiaire ne peut pas gagner : son
rappel est zéro par construction. Le taux de bonnes réponses ne mesure ici que le
déséquilibre du fichier, pas la qualité d'un système ; sur une classe à 0,9 %, il est
saturé avant même qu'on ait commencé à travailler.

**Ce que ça veut dire honnêtement :** avec 8,2 de rappel et 4,0 de précision, notre
modèle n'est pas encore utilisable en production. Mais il est *vrai*, et le 100/96
de la phase 4 ne l'était pas.

---

## Phase 7 Plusieurs témoins, un seul événement

**Doublons exacts.** 362 témoignages sont recopiés mot pour mot sur plusieurs
lignes du fichier des lignes strictement identiques, pas des témoins
différents qui décriraient la même scène avec leurs propres mots.
**Décision : supprimées.** Ce ne sont pas des témoins supplémentaires, ce sont des
doublons de saisie. Relevés restants : **88 513** (88 875 − 362).

**Règle pour reconnaître un même événement :** deux relevés parlent du même
événement s'ils partagent la même ville (`city`), le même état (`state`) et le
même jour d'observation (`datetime` tronqué au jour). Limité aux relevés qui ont
une ville connue (une ville vide ne peut identifier aucun lieu).

| | |
|---|---|
| Événements signalés par plus d'un témoin | **2 399** |
| Témoins pour le plus gros événement | **56** |
| Relevés à cheval train/test dans la découpe d'hier (aléatoire) | **2 362** |

Le plus gros événement est bien celui que le Conseil soupçonnait :
`tinley park \| il \| 2004-10-31`, 56 témoins entre 18h55 et 22h00 le même soir.
Affiché à l'écran, tous alignés, tous du même côté dans la nouvelle découpe.

2 362 relevés étaient à cheval hier entre train et test rien que pour la
soirée de Tinley Park et les événements analogues le modèle de la phase 4/5 a
donc bien triché en partie en reconnaissant des soirées déjà lues, pas en
détectant des canulars.

**Nouvelle découpe :** groupée par `event_id` (`GroupShuffleSplit`, 75/25, seed
42) un événement entier part toujours du même côté.

| | Sur 100 canulars, attrapés (rappel) | Sur 100 signalés, vrais (précision) |
|---|---|---|
| Découpe aléatoire (phase 5) | 8,2 | 4,0 |
| **Découpe par événement** | **9,8** | **5,4** |

Le chiffre remonte légèrement : la découpe aléatoire coupait certains événements
en deux d'une façon qui, en moyenne, pénalisait le modèle plus qu'elle ne
l'aidait (le petit gain de triche sur les 2 362 relevés à cheval était noyé
dans le bruit d'un jeu de test différent). Ce n'est pas la baisse qu'on
attendait partout, et c'est bien le fond du message : la bonne méthode ne
garantit pas un chiffre plus bas, elle garantit un chiffre honnête.

---

## Phase 8 L'ordre des choses

Deux dates existent dans la transmission : `datetime` (quand le témoin a levé
les yeux) et `date_posted` (quand le Bureau a reçu/publié le dossier). **On
coupe sur `date_posted`** : c'est la date à laquelle un relevé devient
réellement disponible pour un système en production. Un témoignage peut
raconter une observation ancienne mais n'arriver au Bureau que des années plus
tard (voir `delai_jours`, phase 5) couper sur `datetime` laisserait le
système s'entraîner sur des dossiers que le Bureau n'avait, à la date de
coupure, pas encore reçus. C'est l'inverse de ce qu'on veut interdire.

| | |
|---|---|
| Date de coupure (75ᵉ percentile de `date_posted`) | **2011-10-10** |
| Relevés côté apprentissage (avant la coupure) | **66 008** |
| Relevés côté test (à partir de la coupure) | **22 505** |
| Proportion de canulars, apprentissage | **0,939 %** |
| Proportion de canulars, test | **0,733 %** |

Les deux proportions ne sont **pas égales**. Le taux de canulars signalés a
changé dans le temps la relecture du Bureau (ou la définition même de ce
qui mérite une note `((hoax))`) n'est pas restée constante sur 20 ans de
transmissions.

| | Sur 100 canulars, attrapés (rappel) | Sur 100 signalés, vrais (précision) |
|---|---|---|
| Découpe par événement (phase 7) | 9,8 | 5,4 |
| **Découpe chronologique** | **10,9** | **3,2** |

---

## Phase 9 Les cases vides

Trois colonnes les plus trouées (sur le fichier dédoublonné, 88 513 relevés) :

| Colonne | % canular si trou | % canular si rempli |
|---|---|---|
| `country` (12 524 trous) | 1,214 % | 0,833 % |
| `state` (7 487 trous) | 1,282 % | 0,850 % |
| `duration_hours_min` (3 095 trous) | 2,294 % | 0,836 % |

Sur les trois colonnes, un relevé troué a **plus** de chances d'être un
canular qu'un relevé rempli parfois presque le double. Le trou porte donc
un vrai signal : ce n'est pas rien, ce sont probablement des dossiers bâclés
ou jamais complétés, exactement le genre de dossier qu'un canular traverse
sans qu'on s'y attarde.

**Traitement retenu :** chaque trou reste sa propre catégorie (`"inconnu"`,
déjà fait depuis la phase 4) et on ajoute un indicateur booléen
`<colonne>_manquant` par colonne trouée. Boucher le trou avec la valeur la
plus fréquente et ne rien ajouter d'autre aurait effacé exactement le signal
qu'on vient de mesurer ; l'indicateur le garde vivant même après imputation.

| | Sur 100 canulars, attrapés (rappel) | Sur 100 signalés, vrais (précision) |
|---|---|---|
| Découpe chronologique (phase 8) | 10,9 | 3,2 |
| **Avec indicateurs de trou** | **10,9** | **3,2** |

Chiffre quasi inchangé (précision +0,04 point, invisible à cette résolution) :
l'information des trous était déjà partiellement captée par les colonnes
`shape`/`state`/`country` elles-mêmes (`"inconnu"` étant déjà sa propre
catégorie). L'indicateur explicite ne fait pas de mal, il rend le signal
disponible sans reposer sur la coïncidence de l'encodage `OneHotEncoder`.

---

## Phase 10 La chaîne de traitement du Bureau

`preparer_honnete_base()` ne calcule que des constantes fixes (`0`, `-1`,
`"inconnu"`) rien n'est appris sur les données avant la découpe. Tous les
calculs appris (vocabulaire TF-IDF, moyennes/écarts-types du `StandardScaler`,
catégories de l'`OneHotEncoder`) vivent à l'intérieur du `Pipeline`
scikit-learn, qui n'est `fit()` que sur la partie apprentissage à chaque appel
de `evaluer()` (`modele.fit(xtr, ytr)`). Aucune étape n'est calculée sur le
fichier entier puis coupée après coup.

**Vérification que le test n'est pas vide de canulars :**

| | |
|---|---|
| Proportion de canulars, apprentissage | 0,939 % |
| Proportion de canulars, test | 0,733 % |

Aucune des deux n'est proche de zéro : à 0,9 % la découpe chronologique
aurait pu, par malchance, donner une partie test presque sans canular ce
n'est pas le cas ici, les deux nombres restent comparables.

**Un relevé inventé traverse toute la chaîne en un seul appel :**

```
{"texte_temoin": "A bright light moved silently across the sky for about a
 minute.", "shape": "light", "state": "ca", "country": "us",
 "duration_seconds": 60.0, "latitude": 34.0, "longitude": -118.0,
 "heure": 22, "mois": 7, "annee": 2005}
    -> modele.predict(...)  -> prediction = 0 (probabilité canular = 0,001)
```

Une seule ligne de code (`modele.predict(ligne)`), aucune étape retapée à la
main : le TF-IDF, la mise à l'échelle et l'encodage des catégories se rejouent
automatiquement, avec les statistiques apprises sur l'apprentissage.

---

## Phase 11 Combien de temps ça a duré

Deux colonnes de durée : `duration_seconds` (censée être un nombre de
secondes propre) et `duration_hours_min` (ce que le témoin a écrit à la
main). Le service de transmission a fabriqué la première à partir de la
seconde et l'a parfois ratée.

**Traitement :** on parse `duration_hours_min` (secondes/minutes/heures,
plages "1-2 minutes", tournures "less than a minute", "1/2 hour", etc.) et on
préfère ce résultat quand il est utilisable ; sinon on retombe sur
`duration_seconds` si elle est non nulle. Aucune ligne n'est supprimée.

| | |
|---|---|
| Relevés dont la durée reste inutilisable après traitement | **7 119** |
| Relevés où les deux colonnes se contredisent | **395** |
| Durée médiane | **180 s** (3 minutes) |
| Relevés annonçant plus d'une journée d'observation | **179** |

Exemple de contradiction affiché : `duration_seconds = 20` mais
`duration_hours_min = "1/2 hour"` (≈ 1 800 s) le service de transmission a
laissé passer un facteur 90.

**Trois durées les plus longues :** 97 836 000 s (`"31 years"`),
82 800 000 s (`"23000hrs"`), 66 276 000 s (`"21 years"`). Manifestement des
témoins qui répondent "toute ma vie" ou une faute de saisie, pas des
observations. **Décision :** on garde la ligne (aucune suppression) mais on
plafonne la durée utilisée par le modèle à 86 400 s (1 jour), avec un
indicateur `duree_aberrante` qui garde la trace du plafonnage plutôt que de
la faire disparaître silencieusement.

| | Sur 100 canulars, attrapés (rappel) | Sur 100 signalés, vrais (précision) |
|---|---|---|
| Indicateurs de trou (phase 9) | 10,9 | 3,2 |
| **Durée récupérée depuis le texte** | **10,9** | **3,3** |

---

## Phase 12 La ville et l'heure

**Les formes (`shape`).** 29 formes distinctes, dont deux paires qui désignent
la même chose sous deux orthographes (`changing`/`changed`,
`circle`/`round`) et six formes signalées moins de 10 fois sur 88 513 relevés
(`delta`, `crescent`, `pyramid`, `flare`, `hexagon`, `dome`), regroupées dans
`other`. **Formes distinctes après traitement : 21** (29 → 21).

**L'heure.** Encodage cyclique (`sin`/`cos` de l'heure sur 24h) plutôt qu'un
entier brut :

| | |
|---|---|
| Distance encodée 23h ↔ 0h | **0,261** |
| Distance encodée 23h ↔ 20h | **0,765** |

23h ressort bien plus proche de 0h que de 20h (contre 1h d'écart vs 21h
d'écart sur l'échelle brute 0-23) : le passage de minuit ne casse plus rien.

**La ville.** 21 981 villes distinctes, dont **14 165** n'apparaissent
qu'une seule fois dans toute la transmission. Pas de colonne par ville : la
largeur du tableau exploserait de 14 à **21 994 colonnes** (13 colonnes
utiles + 1 pour la ville, moins 1, plus 21 981). À la place : un **encodage
cible lissé** le taux de canulars observé pour chaque ville, appris
**uniquement sur la partie apprentissage**, ramené vers la moyenne globale
pour les villes rares (lissage `m = 500`). Largeur réelle du tableau final :
**16 colonnes**.

| | Sur 100 canulars, attrapés (rappel) | Sur 100 signalés, vrais (précision) |
|---|---|---|
| Durée récupérée (phase 11) | 10,9 | 3,3 |
| **Ville (encodage cible) + heure cyclique** | **4,8** | **1,8** |

**Ce chiffre baisse, et il faut le dire clairement.** L'encodage de ville
règle le problème de largeur annoncé par le Conseil, mais il coûte du rappel.
Cause identifiée en testant plusieurs lissages (`m = 50` → 1,2/0,4 ;
`m = 500` → 4,8/1,8 ; `m = 5000` → 4,8/1,8) : un taux appris par ville sur la
période d'apprentissage est bruyant pour les villes rares, et il généralise
mal à la période de test (phase 8), qui couvre des années plus tardives avec
en partie des villes différentes. Le modèle apprend en partie le hasard
d'échantillonnage d'une ville plutôt qu'un vrai signal transférable dans le
temps. Un lissage plus fort (`m = 500`) limite la casse sans l'annuler.
C'est un chiffre honnête qui descend pour une bonne raison, pas une triche
qu'on vient de retirer mais un rappel que corriger un problème de méthode
(la largeur du tableau) peut en révéler un autre (la ville ne transfère pas
bien dans le temps).

---

## Récapitulatif les deux nombres, phase par phase

| Phase | Rappel | Précision | Ce qui a changé |
|---|---|---|---|
| 5 (découpe aléatoire) | 8,2 | 4,0 | Colonnes de fuite retirées (note du Bureau, `date_posted`, délai) |
| 7 (découpe par événement) | 9,8 | 5,4 | Un même événement (plusieurs témoins) ne traverse plus la coupe |
| 8 (découpe chronologique) | 10,9 | 3,2 | Le test est désormais strictement postérieur à l'apprentissage |
| 9 (indicateurs de trou) | 10,9 | 3,2 | Les cases vides gardent leur trace au lieu d'être effacées |
| 11 (durée récupérée) | 10,9 | 3,3 | La durée manquante est récupérée depuis le texte du témoin |
| 12 (ville + heure cyclique) | 4,8 | 1,8 | Largeur du tableau maîtrisée, mais l'encodage de ville généralise mal dans le temps |

Aucun de ces six chiffres n'est celui de la phase 4 (100,0 / 96,1). C'est
voulu : celui-là était faux. Ceux-ci sont honnêtes, et le dernier d'entre eux
raconte une histoire précise qu'on peut défendre devant le Conseil plutôt
qu'un score qu'on ne comprend pas.

---

## Phase 13 La facture du Bureau

Grille votée par le Conseil : un canular raté coûte **30 crédits**, une fausse
alerte **2 crédits**, une bonne réponse ne coûte rien. Un canular raté coûte
donc **15 fois** plus cher qu'une fausse alerte le score par défaut de la
bibliothèque (frontière à 0,5) n'a jamais entendu parler de cette grille.

| Frontière | Facture (crédits) |
|---|---|
| 0,05 | 12 468 |
| 0,10 | 9 260 |
| 0,20 | 7 108 |
| 0,30 | 6 246 |
| 0,40 | 5 838 |
| **0,50 (bibliothèque)** | **5 600** |
| 0,60 | 5 412 |
| 0,70 | 5 282 |
| 0,80 | 5 228 |
| 0,90 | 5 104 |

**L'optimum mathématique pur, en testant toutes les frontières possibles (pas
une grille grossière) : ne signaler personne.** Facture : **4 950 crédits**,
moins chère que n'importe quelle frontière du tableau ci-dessus. C'est un
résultat dérangeant qu'il faut dire clairement : avec seulement 165 canulars
dans le test et une précision aussi faible sur toute la plage de score, la
grille de coûts elle-même dit qu'il est moins cher de ne jamais rien signaler
que de faire tourner le système. **Ce n'est pas la frontière retenue** cette
grille ne facture que les erreurs mesurées sur ce jeu de test, elle ne compte
pas le coût de ne plus avoir de système de détection du tout, qui est
exactement ce que le Bureau nous a demandé de construire. Un système qui ne
signale jamais rien n'est pas une décision défendable, même s'il est le
moins cher sur le papier.

**Frontière retenue :** la moins chère *parmi celles qui attrapent au moins
un canular*. Seuil = **0,9999**, 18 relevés signalés, facture = **4 954
crédits**.

| | Facture |
|---|---|
| Frontière à 0,5 (bibliothèque) | 5 600 crédits |
| **Frontière retenue (0,9999)** | **4 954 crédits** |
| **Écart** | **646 crédits économisés** sur 22 505 relevés de test |

La frontière retenue est **plus sévère** que 0,5, pas plus permissive : elle
signale moins de monde. Ça contredit l'intuition naïve (« un canular raté
coûte plus cher, donc il faut baisser la barre »), et c'est justement parce
que le modèle est trop peu précis à ce niveau : chaque frontière plus basse
ajoute beaucoup plus de fausses alertes qu'elle n'attrape de vrais canulars,
et à 2 crédits la fausse alerte, ce volume finit par coûter plus cher que les
quelques canulars supplémentaires attrapés ne rapportent.

---

## Phase 14 Une promesse à 80 %

Le système trie plutôt qu'il ne chiffre. Dix tranches à effectif égal
(~2 250 relevés chacune), probabilité annoncée contre proportion réellement
observée :

| Tranche | n | Proba annoncée | Taux réel |
|---|---|---|---|
| 0 | 2 251 | 0,06 % | 0,44 % |
| 1 | 2 250 | 0,20 % | 0,31 % |
| 2 | 2 251 | 0,37 % | 0,31 % |
| 3 | 2 250 | 0,62 % | 0,49 % |
| 4 | 2 251 | 0,98 % | 0,40 % |
| 5 | 2 250 | 1,53 % | 0,71 % |
| 6 | 2 250 | 2,44 % | 0,80 % |
| 7 | 2 251 | 4,22 % | 1,02 % |
| 8 | 2 250 | 8,62 % | 1,16 % |
| **9** | 2 251 | **34,59 %** | **1,69 %** |

**Le système est trop confiant.** Sur la tranche la plus suspecte (tranche
9), il annonce en moyenne 34,6 % de chances de canular pour un taux réel de
1,7 % vingt fois moins. La conseillère avait raison : le classement est
globalement dans le bon sens (le taux réel augmente bien de tranche en
tranche), mais le chiffre affiché ne veut rien dire pris isolément.

**Correction :** recalibrage sigmoïde (Platt scaling), appris par validation
croisée **sur la partie apprentissage seule**, jamais sur le test.

| Tranche | n | Proba annoncée | Taux réel |
|---|---|---|---|
| 0 | 2 251 | 0,24 % | 0,31 % |
| 1 | 2 250 | 0,39 % | 0,49 % |
| 2 | 2 251 | 0,49 % | 0,22 % |
| 3 | 2 250 | 0,59 % | 0,53 % |
| 4 | 2 251 | 0,70 % | 0,22 % |
| 5 | 2 250 | 0,82 % | 0,80 % |
| 6 | 2 250 | 0,97 % | 0,89 % |
| 7 | 2 251 | 1,18 % | 0,71 % |
| 8 | 2 250 | 1,55 % | 1,47 % |
| **9** | 2 251 | **4,37 %** | **1,69 %** |

La tranche la plus suspecte passe de 34,6 % annoncé à 4,4 % un facteur 8
de moins, pour le même taux réel de 1,7 %. Toujours un peu optimiste, mais
d'un ordre de grandeur défendable devant le Conseil. **Toutes les
probabilités utilisées à partir d'ici (phases 15 à 17) sont les probabilités
recalibrées.**

---

## Phase 15 Deux analystes, deux chiffres

| | |
|---|---|
| Taille de la partie test | **22 505** |
| Canulars réellement présents dans le test | **165** |

Avec seulement 165 canulars, le rappel et la précision se calculent sur une
poignée de relevés. **2 000 rééchantillonnages bootstrap** de la partie test
(probabilités recalibrées, frontière à 0,5) :

| | Intervalle (95 %) |
|---|---|
| Rappel | **[0,0 % ; 2,0 %]** |
| Précision | **[0,0 % ; 9,1 %]** |

Le recalibrage de la phase 14 a aussi changé la lecture à 0,5 : la précision
et le rappel « bruts » à ce seuil tombent désormais autour de 0,6 % / 2,7 %
(contre 4,8 % / 1,8 % avant recalibrage) preuve supplémentaire, en plus du
bootstrap, que ces chiffres nus bougent beaucoup selon ce qu'on vient de leur
faire subir.

**Réponse au Conseil :** avec 165 canulars dans le test, déplacer 2 ou 3
d'entre eux d'un côté à l'autre d'une découpe fait bouger le chiffre de
plusieurs points sans que le modèle ait changé d'un iota. 0,31 et 0,34
tombent tous les deux dans la même fourchette bootstrap ci-dessus les deux
analystes sont **statistiquement indiscernables** avec cette taille de test,
la question du Conseil n'a pas de réponse tranchée, et le dire est la bonne
réponse.

---

## Phase 16 Trois dossiers sur le Bureau

Trois relevés du test, choisis pour ce qu'ils illustrent, avec deux niveaux
d'explication : le dossier (pourquoi *ce* relevé) et l'ensemble (sur quoi le
système s'appuie globalement).

**1. Marqué canular, forte confiance** (proba = 1,000, vrai label : honnête)
`2004-04-11 22:30`, ville inconnue « *Saw UFO enter "Worm-hole."* »

| Colonne | Contribution | Sens |
|---|---|---|
| `ville_taux_canular` | +15,045 | canular |
| mot « hole » (TF-IDF) | +0,922 | canular |
| `state_manquant` | +0,822 | canular |
| `latitude` | −0,773 | honnête |
| `longitude` | −0,695 | honnête |
| mot « saw » (TF-IDF) | +0,488 | canular |

**2. Tout juste au-dessus de la frontière** (proba = 1,000, vrai label :
canular) `2006-01-20`, ville inconnue « *DON'T KNOW IF THIS WAS A DREAM OR
NOT...* »

| Colonne | Contribution | Sens |
|---|---|---|
| `ville_taux_canular` | +15,045 | canular |
| `shape` = inconnu | −2,080 | honnête |
| `state_manquant` | +0,822 | canular |
| `latitude` / `longitude` | −0,773 / −0,695 | honnête |
| mot « ship » (TF-IDF) | +0,685 | canular |

**3. Canular laissé passer** (proba = 0,795, sous la frontière à 0,9999)
`2013-06-29`, Indianapolis « *Massive cylinder shaped object, thousands of
feet long.* »

| Colonne | Contribution | Sens |
|---|---|---|
| mot « massive » (TF-IDF) | +1,347 | canular |
| « of feet » (TF-IDF) | +1,150 | canular |
| `country` = us | −0,986 | honnête |
| `annee` | +0,980 | canular |
| `state` = in | +0,705 | canular |

**Ce qui saute aux yeux :** les dossiers 1 et 2 partagent exactement la même
raison ville inconnue. Notre encodage cible de la ville (phase 12) a appris
un taux de canulars très élevé pour le compartiment « ville inconnue »
(beaucoup de canulars du fichier d'entraînement ont justement une ville
vide), et ce seul fait pousse à lui seul n'importe quel relevé sans ville
vers 100 % de confiance, à tort une fois sur deux ici. Le dossier 3 est
d'une autre nature : ni la ville ni l'encodage ne sont en cause, c'est
simplement un texte trop neutre (« massive », « thousands of feet ») pour
que le modèle s'alarme suffisamment le raisonnement qui rate ce dossier
n'est pas celui qui fait basculer les deux premiers.

**Classement global (importance par permutation, chute d'*average
precision* quand la colonne est mélangée) :**

| Colonne | Importance |
|---|---|
| `texte_temoin` | +0,0055 |
| `state_manquant` | +0,0008 |
| `country` | +0,0005 |
| `heure_sin` | +0,0005 |
| `annee` | +0,0004 |
| `duration_seconds` | +0,0003 |
| `mois` | +0,0002 |
| `state`, `duree_aberrante`, indicateurs de trou | ≈ 0,0000 |
| `latitude` | −0,0001 |
| `ville_taux_canular` | −0,0002 |
| `longitude` | −0,0004 |
| `shape` | −0,0020 |

**La colonne qui surprend : `ville_taux_canular`.** Elle a le coefficient le
plus fort dans deux des trois dossiers ci-dessus, mais elle n'arrive qu'en
**14ᵉ position sur 16** dans le classement global, avec une importance quasi
nulle (légèrement négative). Elle est décisive pour une poignée de relevés
(ceux sans ville, ou ceux d'une ville au taux d'apprentissage extrême) mais
elle n'aide quasiment pas le modèle sur l'ensemble du test coefficient
énorme, portée réelle minuscule. C'est cohérent avec la phase 12 : c'est la
même colonne dont on avait déjà mesuré qu'elle coûtait du rappel une fois
généralisée à une période plus tardive.

---

## Phase 17 L'angle mort du Bureau

| Zone | n (test) | % canulars | Rappel | Précision |
|---|---|---|---|---|
| États-Unis | 18 909 | 0,64 % | 0,0 % | 0,0 % |
| Canada | 697 | 1,15 % | 0,0 % | 0,0 % |
| Royaume-Uni | 221 | 3,17 % | 0,0 % | 0,0 % |
| Autre / inconnu | 2 678 | 1,08 % | 3,4 % | 5,9 % |

Part des États-Unis dans le test : **84,0 %**. Le chiffre global ne dit rien
de ça : à la frontière retenue en phase 13 (0,9999, très sévère), le système
n'attrape strictement **aucun** canular ni aux États-Unis, ni au Canada, ni
au Royaume-Uni le peu qu'il attrape vient de la zone « autre/inconnu »
(relevés sans pays renseigné). La proportion réelle de canulars n'est pas
non plus la même partout (0,64 % aux États-Unis contre 3,17 % au
Royaume-Uni) alors que le modèle a appris sur un fichier écrasé à 84 % par
un seul pays.

**Décision : une seule frontière pour tout le monde**, celle de la phase 13.
Les zones hors États-Unis pèsent quelques centaines à quelques milliers de
relevés, beaucoup trop peu pour mesurer une frontière spécifique de façon
fiable (voir phase 15, où 165 canulars sur 22 505 relevés donnaient déjà une
fourchette bootstrap large) : une frontière calibrée par zone s'y
calibrerait sur du bruit d'échantillonnage, pas sur un vrai écart de
comportement. Ne pas trancher par zone est donc un choix assumé, pas un
oubli mais il faut que le Bureau sache que son système est, de fait, un
système américain qui n'a presque jamais rien vu d'ailleurs.

---

## Phase 18 La transmission d'archive

**La courbe (années avec au moins 30 relevés, période 1947-2014) :**
écart-type de la proportion annuelle de canulars = **0,62 point**. Elle
n'est pas plate : proche de 0 % avant 1995, elle grimpe et oscille ensuite
entre 1 % et 2,6 % à partir de 2005-2008 (pic à 2,64 % en 2008), avant de
retomber sous 1 % en 2012-2013. Ce n'est pas un hasard d'échantillonnage
(les années récentes comptent des milliers de relevés) : la façon dont le
Bureau annote « hoax » a changé plusieurs fois sur la période, exactement le
scénario que le Conseil décrivait (un chef qui exige la mention, puis on
s'en lasse).

**L'épreuve** apprentissage sur les relevés jusqu'à 2006 (45 581 relevés),
test sur les relevés plus récents (42 932 relevés) :

| | Rappel | Précision |
|---|---|---|
| Phase 8 (découpe chronologique 75/25, coupure 2011-10-10) | 10,9 | 3,2 |
| **Ancien → récent (coupure 2006)** | **2,9** | **4,3** |

Le rappel s'effondre quand on recule la coupure à 2006 : le modèle,
entraîné sur une période où le taux de canulars annotés était bas et
irrégulier (voir la courbe ci-dessus), généralise mal à la période
2007-2008 où le Bureau notait beaucoup plus. C'est la confirmation directe
de ce que la courbe suggérait : le système a appris une définition moyenne
qui ne correspond à aucune année en particulier.

**Ce qu'on surveille en production, sans jamais connaître l'étiquette :**

1. **La proportion de relevés marqués canular par le système**, semaine par
   semaine elle doit rester proche de la proportion historique observée en
   sortie de modèle (pas de la vérité, qu'on n'a pas).
2. **La distribution des probabilités annoncées** (probabilité moyenne, part
   des relevés au-dessus de la frontière) un glissement de cette
   distribution signale que les relevés reçus ne ressemblent plus à ceux de
   l'entraînement, même sans connaître la bonne réponse pour un seul d'entre
   eux.

**Fréquence :** hebdomadaire, alignée sur le rythme d'arrivée des
transmissions. **Seuil de rappel des analystes :** un écart de plus de 2
points de pourcentage sur l'indicateur 1, ou un déplacement de plus de 0,05
de la probabilité moyenne sur l'indicateur 2.
