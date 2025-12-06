# System Architecture

## Overview
The Data Science Recommendation App is a web-based application designed to demonstrate a complete machine learning pipeline for recommendation systems. It allows users to rate items across various categories (colors, movies, politicians, destinations) and uses collaborative filtering algorithms to predict user preferences.

## System Components

### 1. Frontend
- **Technology:** Vanilla JavaScript, HTML5, CSS3.
- **Responsibility:** Handles user interaction, state management, and UI rendering.
- **Key Features:**
    - **Mode Selector:** Allows users to switch between different recommendation categories.
    - **Rating Interface:** Displays items for users to rate on a scale of 1-5.
    - **Leaderboard:** Visualizes the performance of different ML algorithms.
    - **Diagnostics:** Shows confusion matrices and other model performance metrics.
- **State Management:**
    - `localStorage`: Persists `sessionUserId` and `ratingCountsByMode` to maintain state across sessions.
    - In-Memory: Manages `seenItemIdsByMode` to ensure users don't see the same items repeatedly during a session.

### 2. Backend
- **Technology:** Python, FastAPI, SQLAlchemy, SQLite.
- **Responsibility:** API endpoints, data persistence, and machine learning logic.
- **Key Components:**
    - **API Layer:** Exposes RESTful endpoints for the frontend.
    - **Database:** SQLite database (`reco.db`) storing items, users, and ratings.
    - **ML Engine:** Utilizes the `scikit-surprise` library for building and evaluating recommendation models.

### 3. Machine Learning Pipeline
- **Library:** `scikit-surprise`.
- **Algorithms:** Supports various algorithms including SVD, KNNBaseline, NMF, SlopeOne, and CoClustering.
- **Training Process:**
    - **Grid Search:** Performs hyperparameter tuning to find the best configuration for each algorithm.
    - **Evaluation:** Calculates RMSE (Root Mean Squared Error) to rank models.
    - **Prediction:** Generates ratings for unrated items using the best-performing model.

## Data Flow

1.  **Initialization:**
    - Frontend checks for an existing session ID. If not found, it requests a new one from `GET /new-session`.
    - Fetches available modes from `GET /modes`.

2.  **Item Sampling:**
    - Frontend requests items to rate via `GET /sample-items`.
    - Backend returns a random sample of items, excluding those already seen/rated.

3.  **Rating Submission:**
    - User rates an item.
    - Frontend sends the rating to `POST /rate`.
    - Backend saves the rating to the database.

4.  **Model Training & Prediction:**
    - User clicks "Get Recommendations".
    - Frontend calls `POST /train-and-predict`.
    - Backend:
        - Loads all ratings for the current mode.
        - Runs grid search and cross-validation for multiple algorithms.
        - Selects the best model based on RMSE.
        - Predicts ratings for all unrated items for the current user.
        - Returns the leaderboard and top recommendations.

5.  **Visualization:**
    - Frontend renders the leaderboard and recommendation cards.
    - Displays diagnostic data (e.g., confusion matrix) for the best model.
