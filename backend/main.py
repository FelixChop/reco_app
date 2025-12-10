import os
import random
import uuid
from typing import Any, List, Dict, Optional
from pathlib import Path
import json


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, ForeignKey, func
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import pandas as pd
from surprise import (
    Dataset,
    Reader,
    SVD,
    SVDpp,
    KNNBasic,
    KNNWithMeans,
    KNNBaseline,
    NMF,
    SlopeOne,
    BaselineOnly,
    CoClustering,
    NormalPredictor,
)
from surprise.model_selection import GridSearchCV, train_test_split, KFold
from surprise import accuracy

SEED = 42
last_model_wrapper = {"algo": None}

# -------------------------------------------------------------------
# DB setup
# -------------------------------------------------------------------

DB_URL = "sqlite:///./items.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR.parent / "scraping" / "data").resolve()



class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, index=True)
    name = Column(String, index=True)
    subtitle = Column(String, nullable=True)
    hex = Column(String, nullable=True)
    image_keyword = Column(String, nullable=True)
    image_url = Column(String, nullable=True)   # <--- nouveau
    spotify_id = Column(String, nullable=True)  # <--- pour le player
    country = Column(String, nullable=True)     # <--- utile pour destinations (optionnel)



class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    mode = Column(String, index=True)
    item_id = Column(Integer, ForeignKey("items.id"))
    rating = Column(Float)
    is_synthetic = Column(Boolean, default=False)

    item = relationship("Item")


Base.metadata.create_all(bind=engine)

# -------------------------------------------------------------------
# Modes disponibles
# -------------------------------------------------------------------

MODES = {
    "politicians": "Personnalités politiques françaises",
    "colors": "Couleurs",
    "destinations": "Destinations de vacances",
    "movies": "Films",
    "songs": "Musiques",
    "dishes": "Plats Culinaires",
    "books": "Livres",
}

# -------------------------------------------------------------------
# Pydantic schemas
# -------------------------------------------------------------------

class ItemOut(BaseModel):
    id: int
    mode: str
    name: str
    subtitle: Optional[str] = None
    hex: Optional[str] = None
    image_keyword: Optional[str] = None
    image_url: Optional[str] = None      
    spotify_id: Optional[str] = None
    country: Optional[str] = None        

    class Config:
        from_attributes = True



class RatingIn(BaseModel):
    user_id: str
    mode: str
    item_id: int
    rating: int


class PredictionOut(BaseModel):
    item_id: int
    name: str
    subtitle: Optional[str] = None
    hex: Optional[str] = None
    image_keyword: Optional[str] = None
    image_url: Optional[str] = None      
    predicted_rating: float
    average_rating: Optional[float] = None
    vote_count: int = 0
    user_rating: Optional[float] = None



class LeaderboardEntry(BaseModel):
    model_name: str
    rmse: float
    mae: float
    fcp: Optional[float]
    rmse_user: Optional[float]
    best_params: Dict[str, Any]
    rank: int


class TrainResult(BaseModel):
    predictions: List[PredictionOut]
    leaderboard: List[LeaderboardEntry]
    best_model_name: str
    best_model_rmse: float
    best_model_params: Dict[str, Any]
    confusion_matrix: List[List[int]]  # 5x5


class SimilarUsersRequest(BaseModel):
    user_id: str
    mode: str
    limit: int = 5


class ItemRating(BaseModel):
    item_id: int
    name: str
    image_url: Optional[str] = None
    rating: float
    subtitle: Optional[str] = None
    hex: Optional[str] = None
    image_keyword: Optional[str] = None
    predicted_rating: float = 0.0
    average_rating: float = 0.0
    vote_count: int = 0


class CommonItem(BaseModel):
    item_id: int
    name: str
    image_url: Optional[str] = None
    my_rating: float
    their_rating: float
    subtitle: Optional[str] = None
    hex: Optional[str] = None
    image_keyword: Optional[str] = None
    predicted_rating: float = 0.0
    average_rating: float = 0.0
    vote_count: int = 0


class Neighbor(BaseModel):
    neighbor_id: str
    similarity_score: float
    common_items: List[CommonItem]
    other_items: List[ItemRating]
    user_items: List[ItemRating]


class SimilarUsersResponse(BaseModel):
    neighbors: List[Neighbor]


# -------------------------------------------------------------------
# Seed des différents modes
# -------------------------------------------------------------------

COLOR_ITEMS = [
    ("Blue Klein", "#002FA7"),
    ("Calico", "#E0C68C"),
    ("Light Green", "#90EE90"),
    ("Cyan", "#00FFFF"),
    ("Pink", "#FFC0CB"),
    ("Purple", "#800080"),
    ("Metal", "#B0C4DE"),
    ("Coral", "#FF7F50"),
    ("Mint", "#98FF98"),
    ("Lavender", "#E6E6FA"),
    ("Sunset Orange", "#FD5E53"),
    ("Ocean Blue", "#1B3B6F"),
    ("Lemon", "#FFF44F"),
    ("Forest", "#228B22"),
    ("Sky", "#87CEEB"),
    ("Sand", "#C2B280"),
    ("Cherry", "#DE3163"),
    ("Turquoise", "#40E0D0"),
    ("Peach", "#FFDAB9"),
    ("Royal Blue", "#4169E1"),
]

def load_json_safe(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.exists():
        print(f"⚠️ Fichier {path} introuvable, je skippe.")
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def seed_items():
    db = SessionLocal()
    try:
        # On tente de lire la DB. Si le schéma a changé (ex: colonne manquante),
        # cela va lever une OperationalError.
        try:
            count = db.query(Item).count()
        except OperationalError:
            print("⚠️ Schéma de base de données incorrect détecté (colonne manquante ?).")
            print("♻️  Re-création de la base de données...")
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            count = 0


        # On reseed TOUTES les thématiques : pour chaque mode, on supprime les
        # entrées existantes puis on réinsère depuis les données sources.

        # 1) Couleurs (données embarquées)
        print("Reseeding colors...")
        db.query(Item).filter(Item.mode == "colors").delete()
        for name, hex_code in COLOR_ITEMS:
            db.add(Item(
                mode="colors",
                name=name,
                hex=hex_code,
            ))
        
        # 2) Politiciens
        print("Reseeding politicians...")
        db.query(Item).filter(Item.mode == "politicians").delete()
        pol_data = load_json_safe("politicians.json")
        for row in pol_data:
            db.add(Item(
                mode="politicians",
                name=row["name"],
                subtitle=row.get("subtitle") or "",
                image_keyword=row.get("image_keyword") or row["name"] + " portrait",
                image_url=row.get("image_url"),
            ))

        # 3) Destinations
        print("Reseeding destinations...")
        db.query(Item).filter(Item.mode == "destinations").delete()
        dest_data = load_json_safe("destinations.json")
        for row in dest_data:
            db.add(Item(
                mode="destinations",
                name=row["name"],
                subtitle=row.get("subtitle") or row.get("country") or "",
                country=row.get("country"),
                image_keyword=row.get("image_keyword") or (row["name"] + " travel"),
                image_url=row.get("image_url"),
            ))

        # 4) Films
        print("Reseeding movies...")
        db.query(Item).filter(Item.mode == "movies").delete()
        movies_data = load_json_safe("movies.json")
        for row in movies_data:
            poster_path = row.get("poster_path")
            if poster_path:
                tmdb_image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            else:
                tmdb_image_url = None

            db.add(Item(
                mode="movies",
                name=row["name"],
                subtitle=row.get("subtitle") or "",
                image_keyword=row.get("image_keyword") or (row["name"] + " movie poster"),
                image_url=tmdb_image_url,
            ))

        # 5) Musiques
        print("Reseeding songs...")
        db.query(Item).filter(Item.mode == "songs").delete()
        songs_data = load_json_safe("songs.json")
        for row in songs_data:
            db.add(Item(
                mode="songs",
                name=row["name"],
                subtitle=row.get("subtitle") or "",
                image_keyword=row.get("image_keyword") or (row["name"] + " song"),
                image_url=row.get("image_url"),
                spotify_id=row.get("spotify_id"),
            ))
        
        # 6) Plats
        print("Reseeding dishes...")
        db.query(Item).filter(Item.mode == "dishes").delete()
        dishes_data = load_json_safe("dishes.json")
        for row in dishes_data:
            db.add(Item(
                mode="dishes",
                name=row["name"],
                subtitle=row.get("subtitle") or "",
                country=row.get("country"),
                image_keyword=row.get("image_keyword") or (row["name"] + " dish"),
                image_url=row.get("image_url"),
            ))

        # 7) Livres
        print("Reseeding books from books.json...")
        db.query(Item).filter(Item.mode == "books").delete()
        books_data = load_json_safe("books.json")
        for row in books_data:
            db.add(Item(
                mode="books",
                name=row["name"],
                subtitle=row.get("subtitle") or "",
                image_keyword=row.get("image_keyword") or (row["name"] + " book"),
                image_url=row.get("image_url"),
            ))

        db.commit()


        # Seed synthétique des ratings : SUPPRIMÉ sur demande utilisateur
        # On ne garde que les items.
    finally:
        db.close()



seed_items()

# -------------------------------------------------------------------
# Helpers ML
# -------------------------------------------------------------------

def build_surprise_dataset(mode: str):
    db = SessionLocal()
    try:
        real_ratings = db.query(Rating).filter(
            Rating.is_synthetic == False,
            Rating.mode == mode,
        ).all()
        
        if not real_ratings:
             return Dataset.load_from_df(pd.DataFrame(columns=["user", "item", "rating"]), Reader(rating_scale=(1, 5)))

        rows = [{"user": r.user_id, "item": r.item_id, "rating": r.rating} for r in real_ratings]
        df = pd.DataFrame(rows)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
        return data
    finally:
        db.close()


def compute_confusion_matrix(predictions) -> List[List[int]]:
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for p in predictions:
        true_rating = int(round(float(p.r_ui)))
        est = int(round(float(p.est)))
        true_rating = min(5, max(1, true_rating))
        est = min(5, max(1, est))
        matrix[true_rating - 1][est - 1] += 1
    return matrix


def run_grid_search(
    name: str,
    algo_cls,
    param_grid: Dict[str, Any],
    data,
):
    if not param_grid:
        return {}, None

    print(f"🔍 Grid search for {name}…")
    # Use KFold with fixed seed for determinism
    kf = KFold(n_splits=3, random_state=SEED, shuffle=True)
    gs = GridSearchCV(algo_cls, param_grid, measures=["rmse"], cv=kf, joblib_verbose=0)
    gs.fit(data)
    best_params = gs.best_params["rmse"]
    best_score = gs.best_score["rmse"]
    print(f"✅ Best params {name}: {best_params} (cv RMSE={best_score:.4f})")
    return best_params, best_score


def evaluate_algorithm(
    name: str,
    algo_cls,
    param_grid: Dict[str, Any],
    data,
    trainset,
    testset,
    user_id: str,
):
    best_params, best_cv_rmse = run_grid_search(name, algo_cls, param_grid, data)
    algo = algo_cls(**best_params) if best_params else algo_cls()

    try:
        algo.fit(trainset)
        preds = algo.test(testset)
        
        # Calculate all metrics (sans MSE)
        rmse_global = accuracy.rmse(preds, verbose=False)
        mae_global = accuracy.mae(preds, verbose=False)
        try:
            fcp_global = accuracy.fcp(preds, verbose=False)
        except ValueError:
            fcp_global = None

        user_preds = [p for p in preds if p.uid == user_id]
        rmse_user = accuracy.rmse(user_preds, verbose=False) if user_preds else None

        return {
            "model_name": name,
            "rmse": rmse_global,
            "mae": mae_global,
            "fcp": fcp_global,
            "rmse_user": rmse_user,
            "preds": preds,
            "algo": algo,
            "best_params": best_params if best_params is not None else {},
            "best_cv_rmse": best_cv_rmse,
            "failed": False,
        }
    except Exception as e:
        print(f"⚠️ Algorithm {name} failed during fit/test: {e}")
        return {
            "model_name": name,
            "rmse": float('inf'), # Set RMSE to infinity for sorting purposes
            "mae": None,
            "fcp": None,
            "rmse_user": None,
            "preds": [],
            "algo": None,
            "best_params": best_params if best_params is not None else {},
            "best_cv_rmse": best_cv_rmse,
            "failed": True,
        }


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app.mount("/static", StaticFiles(directory=BACKEND_DIR / "static"), name="static")



@app.get("/modes")
def get_modes():
    """Renvoie les modes disponibles."""
    return [{"id": k, "label": v} for k, v in MODES.items()]


@app.get("/new-session", response_model=str)
def new_session():
    return str(uuid.uuid4())


@app.get("/sample-items", response_model=List[ItemOut])
def sample_items(
    mode: str = Query("colors"),
    limit: int = Query(3, ge=1, le=50),
    exclude_ids: List[int] = Query(default_factory=list),
):
    if mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")
    db = SessionLocal()
    try:
        query = db.query(Item).filter(Item.mode == mode)
        if exclude_ids:
            query = query.filter(~Item.id.in_(exclude_ids))

        available_items = query.all()
        if len(available_items) == 0:
            return []

        sample_size = min(limit, len(available_items))
        return random.sample(available_items, sample_size)
    finally:
        db.close()


@app.post("/rate")
def rate_item(payload: RatingIn):
    if payload.mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")
    db = SessionLocal()
    try:
        item = db.query(Item).filter(
            Item.id == payload.item_id,
            Item.mode == payload.mode,
        ).first()
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        # Update or create rating
        rating = db.query(Rating).filter(
            Rating.user_id == payload.user_id,
            Rating.item_id == payload.item_id,
            Rating.mode == payload.mode
        ).first()

        if rating:
            rating.rating = float(payload.rating)
            rating.is_synthetic = False
        else:
            rating = Rating(
                user_id=payload.user_id,
                mode=payload.mode,
                item_id=payload.item_id,
                rating=float(payload.rating),
                is_synthetic=False,
            )
            db.add(rating)
            
        db.commit()
        return {"status": "ok"}
    finally:
        db.close()


@app.get("/rating-counts")
def get_rating_counts(user_id: str):
    db = SessionLocal()
    try:
        counts = (
            db.query(Rating.mode, func.count(Rating.id))
            .filter(
                Rating.user_id == user_id,
                Rating.is_synthetic == False,  # noqa: E712
            )
            .group_by(Rating.mode)
            .all()
        )
        return {mode: count for mode, count in counts}
    finally:
        db.close()


@app.get("/voter-count")
def get_voter_count(mode: str):
    if mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")
    db = SessionLocal()
    try:
        count = db.query(Rating.user_id).filter(
            Rating.mode == mode,
            Rating.is_synthetic == False
        ).distinct().count()

        total_items = db.query(Item).filter(Item.mode == mode).count()

        return {"mode": mode, "count": count, "total_items": total_items}
    finally:
        db.close()


@app.post("/train-and-predict", response_model=TrainResult)
def train_and_predict(user_id: str, mode: str):
    if mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")

    db = SessionLocal()
    try:
        data = build_surprise_dataset(mode)
        # If dataset is too small, we might get errors in train_test_split
        # But with hybrid logic we ensure ~100 users, so it should be fine.
        
        trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)

        search_spaces = {
            # SVD: Lower regularization to allow fitting to outliers (1s and 5s)
            "SVD": {"n_epochs": [20, 30], "lr_all": [0.005, 0.01], "reg_all": [0.01, 0.05], "random_state": [SEED]},
            "SVDpp": {"n_epochs": [10, 20], "lr_all": [0.005, 0.01], "random_state": [SEED]},
            # NMF: Lower regularization
            "NMF": {"n_factors": [10, 20], "n_epochs": [30, 50], "reg_pu": [0.01, 0.05], "reg_qi": [0.01, 0.05], "random_state": [SEED]},
            # KNN: Lower k to capture local similarities (less smoothing)
            "KNNBasic": {"k": [1,2,3,4,5, 10, 20], "sim_options": {"name": ["cosine", "msd"], "user_based": [True, False]}},
            "KNNWithMeans": {"k": [1,2,3,4,5, 10, 20], "sim_options": {"name": ["cosine", "pearson"], "user_based": [True, False]}},
            "KNNBaseline": {
                "k": [1,2,3,4,5, 10, 20],
                "bsl_options": {"method": ["als", "sgd"], "reg_i": [1, 5], "reg_u": [1, 5]},
                "sim_options": {"name": ["cosine"], "user_based": [True, False], "min_support": [1, 2]},
            },
            "SlopeOne": {},
            "BaselineOnly": {"bsl_options": {"method": ["als", "sgd"], "reg_i": [1, 5], "reg_u": [1, 5]}},
            # CoClustering: Smaller clusters for smaller data
            "CoClustering": {"n_cltr_u": [2, 3], "n_cltr_i": [2, 3], "n_epochs": [20, 30], "random_state": [SEED]},
            "NormalPredictor": {},
        }

        algo_classes = {
            "SVD": SVD,
            "SVDpp": SVDpp,
            "NMF": NMF,
            "KNNBasic": KNNBasic,
            "KNNWithMeans": KNNWithMeans,
            "KNNBaseline": KNNBaseline,
            "SlopeOne": SlopeOne,
            "BaselineOnly": BaselineOnly,
            "CoClustering": CoClustering,
            "NormalPredictor": NormalPredictor,
        }

        algo_results: List[Dict[str, Any]] = []

        for name, algo_cls in algo_classes.items():
            result = evaluate_algorithm(
                name=name,
                algo_cls=algo_cls,
                param_grid=search_spaces.get(name, {}),
                data=data,
                trainset=trainset,
                testset=testset,
                user_id=user_id,
            )
            algo_results.append(result)

        # Custom sorting:
        # 1. RMSE (asc)
        # 2. Complexity (Baselines first): NormalPredictor/BaselineOnly = 0, others = 1
        # 3. User RMSE (asc)
        
        baselines = {"NormalPredictor", "BaselineOnly"}
        
        algo_results.sort(
            key=lambda x: (
                x["rmse"],
                0 if x["model_name"] in baselines else 1,
                x["rmse_user"] if x["rmse_user"] is not None else float("inf"),
            )
        )

        leaderboard: List[Dict[str, Any]] = []
        algo_predictions = {}
        for idx, res in enumerate(algo_results, start=1):
            leaderboard.append(
                {
                    "model_name": res["model_name"],
                    "rmse": res["rmse"],
                    "mae": res["mae"],
                    "fcp": res["fcp"],
                    "rmse_user": res["rmse_user"],
                    "best_params": res["best_params"],
                    "rank": idx,
                }
            )
            algo_predictions[res["model_name"]] = {"algo": res["algo"], "preds": res["preds"]}

        best_model_name = leaderboard[0]["model_name"]
        best_model_rmse = leaderboard[0]["rmse"]
        best_model_params = leaderboard[0]["best_params"]
        best_algo = algo_predictions[best_model_name]["algo"]
        best_preds_test = algo_predictions[best_model_name]["preds"]
        
        # Filter predictions to only include current user's REAL votes (not synthetic)
        # Get user's real rated items
        user_real_ratings = db.query(Rating).filter(
            Rating.user_id == user_id,
            Rating.mode == mode,
            Rating.is_synthetic == False
        ).all()
        user_real_item_ids = {r.item_id for r in user_real_ratings}
        
        # Filter testset predictions to only include user's real votes
        user_preds_test = [
            p for p in best_preds_test 
            if p.uid == user_id and p.iid in user_real_item_ids
        ]
        
        confusion_matrix = compute_confusion_matrix(user_preds_test)

        # Cache the best algo for similar user predictions
        last_model_wrapper["algo"] = best_algo

        # Predictions for user on ALL items (including those already rated, as requested)
        all_items = db.query(Item).filter(Item.mode == mode).all()
        
        # We still want to know which ones were rated to maybe flag them in UI, 
        # but the request says "see elements I rated initially".
        # So we predict for everything.
        
        predictions = []
        for it in all_items:
            # For items the user has rated, we could return the actual rating or the predicted one.
            # Usually 'predict' returns the estimated rating even if the user rated it (unless impossible).
            # Surprise's predict method will return the known rating as r_ui if passed, but here we just want 'est'.
            
            est = best_algo.predict(uid=user_id, iid=it.id).est
            
            # Get stats
            stats = db.query(
                func.avg(Rating.rating).label("avg"),
                func.count(Rating.id).label("count")
            ).filter(
                Rating.item_id == it.id,
                Rating.mode == mode,
                Rating.is_synthetic == False
            ).first()
            
            avg_val = stats.avg if stats.avg is not None else 0.0
            count_val = stats.count if stats.count is not None else 0
            
            # Get user rating if any
            user_r = db.query(Rating).filter(
                Rating.user_id == user_id,
                Rating.item_id == it.id,
                Rating.mode == mode
            ).first()
            user_val = user_r.rating if user_r else None

            predictions.append(
                {
                    "item_id": it.id,
                    "name": it.name,
                    "subtitle": it.subtitle,
                    "hex": it.hex,
                    "image_keyword": it.image_keyword,
                    "image_url": it.image_url,
                    "predicted_rating": float(round(est, 2)),
                    "average_rating": float(round(avg_val, 2)) if avg_val else 0.0,
                    "vote_count": count_val,
                    "user_rating": user_val,
                }
            )

        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)

        return TrainResult(
            predictions=[PredictionOut(**p) for p in predictions],
            leaderboard=[LeaderboardEntry(**e) for e in leaderboard],
            best_model_name=best_model_name,
            best_model_rmse=best_model_rmse,
            best_model_params=best_model_params,
            confusion_matrix=confusion_matrix,
        )
    finally:
        db.close()


@app.post("/similar-users", response_model=SimilarUsersResponse)
def get_similar_users(payload: SimilarUsersRequest):
    import numpy as np
    from scipy.stats import pearsonr
    
    if payload.mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")

    db = SessionLocal()
    try:
        # 1. Fetch all ratings for this mode
        all_ratings = db.query(Rating).filter(
            Rating.mode == payload.mode,
            Rating.is_synthetic == False
        ).all()

        # 2. Build user-item matrix
        user_map = {}  # {user_id: {item_id: rating}}
        for r in all_ratings:
            if r.user_id not in user_map:
                user_map[r.user_id] = {}
            user_map[r.user_id][r.item_id] = r.rating

        if payload.user_id not in user_map:
            return SimilarUsersResponse(neighbors=[])

        my_ratings = user_map[payload.user_id]
        if len(my_ratings) < 3:
            return SimilarUsersResponse(neighbors=[])

        # 3. Compute Pearson correlation with other users
        similarities = []
        my_items = set(my_ratings.keys())

        for uid, u_ratings in user_map.items():
            if uid == payload.user_id:
                continue
            
            if len(u_ratings) < 3:
                continue

            their_items = set(u_ratings.keys())
            common = my_items.intersection(their_items)
            
            if len(common) < 3:  # Need at least 3 common items for meaningful correlation
                continue
            
            # Get ratings for common items
            my_common_ratings = [my_ratings[iid] for iid in common]
            their_common_ratings = [u_ratings[iid] for iid in common]
            
            # Calculate Pearson correlation
            try:
                corr, _ = pearsonr(my_common_ratings, their_common_ratings)
                # Pearson returns -1 to 1, convert to 0 to 1 for display
                # We only want positive correlations (similar users)
                if corr > 0:
                    similarities.append((uid, corr, len(common)))
            except:
                # If correlation fails (e.g., constant ratings), skip
                continue

        # Sort by correlation (descending), then by number of common items
        similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_k = similarities[:payload.limit]

        # 4. Fetch details & build response
        all_items_db = db.query(Item).filter(Item.mode == payload.mode).all()
        item_lookup = {i.id: i for i in all_items_db}

        # Helper model
        best_algo = last_model_wrapper["algo"]

        def get_pred_stats(item_id, target_user_id):
            est = 0.0
            if best_algo:
                est = best_algo.predict(uid=target_user_id, iid=item_id).est
            
            stats = db.query(
                func.avg(Rating.rating).label("avg"),
                func.count(Rating.id).label("count")
            ).filter(
                Rating.item_id == item_id,
                Rating.mode == payload.mode,
                Rating.is_synthetic == False
            ).first()
            
            avg_val = stats.avg if stats.avg is not None else 0.0
            count_val = stats.count if stats.count is not None else 0
            return float(round(est, 2)), float(round(avg_val, 2)), count_val

        # Collect neighbors data
        my_ratings_map = {r.item_id: r.rating for r in db.query(Rating).filter(Rating.user_id == payload.user_id, Rating.mode == payload.mode).all()}
        my_items_ids = set(my_ratings_map.keys())

        neighbors_out = []
        for uid, sim, num_common in top_k:
            their_ratings = db.query(Rating).filter(Rating.user_id == uid, Rating.mode == payload.mode, Rating.is_synthetic == False).all()
            their_ratings_map = {r.item_id: r.rating for r in their_ratings}
            their_items_ids = set(their_ratings_map.keys())
            
            common = my_items_ids.intersection(their_items_ids)
            
            # Build ItemRating lists
            common_list = []
            other_list = []
            user_list = []

            # Common items
            for iid in common:
                it = item_lookup.get(iid)
                if it:
                    pred, avg, cnt = get_pred_stats(iid, payload.user_id)
                    common_list.append(CommonItem(
                        item_id=iid,
                        name=it.name,
                        image_url=it.image_url,
                        my_rating=my_ratings_map[iid],
                        their_rating=their_ratings_map[iid],
                        predicted_rating=pred,
                        average_rating=avg,
                        vote_count=cnt
                    ))

            # They Only (Items I haven't seen)
            for iid in their_items_ids - common:
                it = item_lookup.get(iid)
                if it:
                    pred, avg, cnt = get_pred_stats(iid, payload.user_id)
                    other_list.append(ItemRating(
                        item_id=iid,
                        name=it.name,
                        image_url=it.image_url,
                        rating=their_ratings_map[iid],
                        predicted_rating=pred,
                        average_rating=avg,
                        vote_count=cnt
                    ))
            
            # Me Only (Items they haven't seen)
            for iid in my_items_ids - common:
                it = item_lookup.get(iid)
                if it:
                    pred, avg, cnt = get_pred_stats(iid, payload.user_id)
                    user_list.append(ItemRating(
                        item_id=iid,
                        name=it.name,
                        image_url=it.image_url,
                        rating=my_ratings_map[iid],
                        predicted_rating=pred,
                        average_rating=avg,
                        vote_count=cnt
                    ))
            
            # Anonymize ID
            short_id = f"User {uid[:4].upper()}..."
            
            neighbors_out.append(Neighbor(
                neighbor_id=short_id,
                similarity_score=float(sim),  # Pearson correlation (0 to 1)
                common_items=common_list,
                other_items=other_list,
                user_items=user_list
            ))

        return SimilarUsersResponse(neighbors=neighbors_out)

    finally:
        db.close()

