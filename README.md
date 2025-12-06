# Data Science Recommendation App

A web application demonstrating a complete machine learning recommendation pipeline. Users can rate items across different categories (colors, movies, politicians, destinations), and the system uses collaborative filtering (via the `scikit-surprise` library) to predict preferences and recommend new items.

## Features

- **Multi-Mode Recommendation:** Support for various item categories.
- **Interactive UI:** Clean, responsive interface for rating items and viewing recommendations.
- **Real-time Training:** Models are trained on-the-fly using user ratings.
- **Model Leaderboard:** Compares multiple algorithms (SVD, KNN, NMF, etc.) and selects the best one.
- **Diagnostics:** Visualizes model performance with confusion matrices.
- **Persistent State:** Saves user sessions and ratings.

## Tech Stack

- **Frontend:** Vanilla JavaScript, HTML5, CSS3.
- **Backend:** Python 3.11+, FastAPI.
- **Database:** SQLite, SQLAlchemy.
- **Machine Learning:** `scikit-surprise`, Pandas.

## Project Structure

- `frontend/`: Static web assets (HTML, CSS, JS).
- `backend/`: FastAPI application and ML logic.
- `scraping/`: Data seeding scripts and JSON datasets.
- `docs/`: Technical documentation.

## Setup & Installation

### Prerequisites
- Python 3.11 or higher
- Node.js (optional, for serving frontend if preferred over Python's http.server)

### Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Start the server:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`.

### Frontend Setup

1.  Open `frontend/index.html` directly in your browser, OR serve it using a static file server:
    ```bash
    # Using Python
    cd frontend
    python -m http.server 3000
    ```

2.  Ensure the frontend is configured to talk to the backend. By default, it expects the API at `http://localhost:8000`.

## Usage

1.  **Select a Mode:** Choose a category (e.g., Colors, Movies) from the top menu.
2.  **Rate Items:** Rate at least 3 items to unlock recommendations.
3.  **Get Recommendations:** Click the "Get Recommendations" button.
4.  **View Results:** Explore the recommended items, check the model leaderboard, and view diagnostic data.

## Documentation

- [System Architecture](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)
