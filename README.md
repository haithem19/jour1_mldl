# OVNI — Bureau d'Analyse Terrestre, Partie 1

Analyse des 88 875 relevés de la sonde Klaxo-3.

```bash
pip install -r requirements.txt
python analyse.py
```

Le script télécharge lui-même la transmission (~15 Mo, non versionnée) puis
enchaîne les 6 phases et affiche tous les chiffres du rapport.

- `analyse.py` — le script, du téléchargement au dernier chiffre.
- `RAPPORT.md` — les résultats et les décisions. C'est ce que lit le Conseil.
