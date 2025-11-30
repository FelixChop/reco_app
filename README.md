# Machine Learning Recommendation App

Application web simple qui permet de noter des éléments (couleurs, films, destinations, personnalités) et d'obtenir des recommandations basées sur Surprise (SVD, KNNBaseline, NMF).

## Structure du projet
- `frontend/` : page statique HTML/CSS/JS qui pilote le flux d'expérience utilisateur (notation, calcul des recos, affichage du leaderboard, etc.).
- `backend/` : API FastAPI qui gère les sessions, le stockage SQLite, la collecte des notes et l'entraînement des modèles de recommandation.
- `scraping/` : fichiers de données JSON utilisés pour alimenter la base.
- `docs/` : documentation technique et notes d'analyse.

## Prérequis
- Python 3.11+
- Node/npm (optionnel, l'UI est statique et se sert avec n'importe quel serveur de fichiers)

## Installation backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'API expose :
- `GET /modes` : modes disponibles
- `GET /new-session` : identifiant utilisateur
- `GET /sample-items?mode=<mode>` : tirage de 3 items à noter
- `POST /rate` : enregistrement d'une note `{user_id, mode, item_id, rating}`
- `POST /train-and-predict?user_id=<id>&mode=<mode>` : entraînement + recommandations + leaderboard

## Utilisation du frontend
Ouvre `frontend/index.html` dans un navigateur ou sers le dossier `frontend/` via un serveur statique (ex. `python -m http.server 3000`).
Assure-toi que l'API tourne sur `http://localhost:8000` ou ajuste `API_BASE` dans `frontend/app.js`.

## Tests
Aucun test automatisé n'est fourni. La vérification principale consiste à :
1. Lancer le backend (voir ci-dessus).
2. Ouvrir le frontend et noter au moins 3 éléments.
3. Cliquer sur « Obtenir mes recommandations » et vérifier l'affichage des cartes, du leaderboard et de la matrice de confusion.

## Notes importantes
- La base SQLite est créée au lancement et seedée à partir des fichiers du dossier `scraping/data`.
- L'interface garde l'identifiant utilisateur en `localStorage` (`color_recommender_user_id`).
- Les images s'appuient sur les URLs renvoyées par l'API ou sur un fallback Unsplash basé sur le mot-clé.
