# Audit rapide des régressions

## Symptômes observés
- Les boutons n'avaient plus de styles et certains contrôles étaient invisibles.
- Les éléments à noter ne s'affichaient plus et le script se terminait tôt avec des erreurs JavaScript.
- Des actions (passer, recommencer, trier) ne déclenchaient rien ou utilisaient des variables inexistantes.

## Causes principales
1. **Incohérences DOM ↔️ JavaScript** :
   - Le script référençait des éléments absents (`mode-select`) ou mal nommés (`ratingStarsContainer`, `sessionId`, `currentItem`).
   - Plusieurs fonctions étaient dupliquées (`fetchModes`, `sendRating`) avec des signatures différentes, ce qui écrasait le comportement initial.
2. **État applicatif incomplet** :
   - Aucun suivi clair de l'item courant ni de l'indice, rendant impossible le passage à l'item suivant.
   - Le compteur de notes n'était jamais remis à zéro lors d'un changement de mode.
3. **Styles obsolètes** :
   - Les classes utilisées dans le HTML (`primary-btn`, `secondary-btn`, `primary-outline-btn`) n'étaient pas définies dans la feuille de style actuelle.

## Correctifs appliqués
- Refonte de `frontend/app.js` pour reconstruire un flux linéaire : initialisation de session, sélection de mode via des onglets, tirage des items, saisie des notes, calcul des recommandations, puis affichage des diagnostics.
- Harmonisation des noms d'éléments DOM et des variables d'état (ex. `currentItem`, `starContainer`, `sessionUserId`).
- Ajout d'un fallback Unsplash lorsque l'API ne fournit pas d'image.
- Réactivation du compteur de notes et du bouton « Obtenir mes recommandations » après 3 votes.
- Amélioration des composants d'affichage (cartes de recommandations, leaderboard, matrice de confusion) pour qu'ils restent synchronisés avec les données retournées par l'API.
- Rafraîchissement complet des styles pour les boutons, cartes et onglets de mode.

## Pistes d'amélioration futures
- Ajouter des tests end-to-end (Playwright) pour vérifier le flux complet (3 notes → calcul → affichage des cartes et diagnostics).
- Exposer un endpoint de santé (`/health`) dans le backend pour faciliter le monitoring.
- Gérer les erreurs réseau côté frontend avec des messages contextualisés (ex. bannière d'erreur persistante plutôt qu'un simple `alert`).
- Ajouter un linter/formatter (Prettier + ESLint) et un pipeline CI simple (lint + `python -m compileall backend`).
