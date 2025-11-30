import os
import random
import uuid
from typing import List, Dict, Optional
from pathlib import Path
import json


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBaseline, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy

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



class LeaderboardEntry(BaseModel):
    model_name: str
    rmse: float
    rank: int


class TrainResult(BaseModel):
    predictions: List[PredictionOut]
    leaderboard: List[LeaderboardEntry]
    best_model_name: str
    best_model_rmse: float
    confusion_matrix: List[List[int]]  # 5x5


# -------------------------------------------------------------------
# Seed des différents modes
# -------------------------------------------------------------------
# NB: j’en mets une vingtaine par catégorie pour l’exemple.
# Tu peux étendre facilement à 50/100/100/100 en copiant le pattern.

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

POLITICIAN_ITEMS = [
    ("Emmanuel Macron", "Président", "Emmanuel Macron portrait"),
    ("Jean-Luc Mélenchon", "LFI", "Jean-Luc Mélenchon portrait"),
    ("Marine Le Pen", "RN", "Marine Le Pen portrait"),
    ("Édouard Philippe", "Horizons", "Édouard Philippe portrait"),
    ("François Hollande", "PS", "François Hollande portrait"),
    ("Nicolas Sarkozy", "LR", "Nicolas Sarkozy portrait"),
    ("Bruno Le Maire", "Ministre", "Bruno Le Maire portrait"),
    ("Gérald Darmanin", "Ministre", "Gérald Darmanin portrait"),
    ("Olivier Faure", "PS", "Olivier Faure portrait"),
    ("Jordan Bardella", "RN", "Jordan Bardella portrait"),
    ("Valérie Pécresse", "Île-de-France", "Valérie Pécresse portrait"),
    ("Anne Hidalgo", "Maire de Paris", "Anne Hidalgo portrait"),
    ("Fabien Roussel", "PCF", "Fabien Roussel portrait"),
    ("Yannick Jadot", "EELV", "Yannick Jadot portrait"),
    ("Raphaël Glucksmann", "PS/Place Publique", "Raphaël Glucksmann portrait"),
]

DESTINATION_ITEMS = [
    ("Paris", "France", "Paris skyline"),
    ("Bordeaux", "France", "Bordeaux city"),
    ("Lyon", "France", "Lyon city"),
    ("Nice", "France", "Nice French Riviera"),
    ("Strasbourg", "France", "Strasbourg Christmas market"),
    ("Marseille", "France", "Marseille old port"),
    ("Chamonix", "France", "Chamonix Mont Blanc"),
    ("Santorini", "Grèce", "Santorini island"),
    ("Bali", "Indonésie", "Bali beach"),
    ("New York", "USA", "New York skyline"),
    ("Tokyo", "Japon", "Tokyo city night"),
    ("Lisbonne", "Portugal", "Lisbon streets"),
    ("Rome", "Italie", "Rome Colosseum"),
    ("Barcelone", "Espagne", "Barcelona Sagrada Familia"),
    ("Séville", "Espagne", "Seville city"),
    ("Reykjavik", "Islande", "Iceland northern lights"),
    ("Sydney", "Australie", "Sydney Opera House"),
    ("Cape Town", "Afrique du Sud", "Cape Town Table Mountain"),
    ("Marrakech", "Maroc", "Marrakech market"),
    ("Québec", "Canada", "Quebec old town"),
]

MOVIE_ITEMS = [
    ("The Shawshank Redemption", "1994", "movie poster jail drama"),
    ("The Godfather", "1972", "mafia movie poster"),
    ("The Dark Knight", "2008", "batman movie poster"),
    ("Pulp Fiction", "1994", "pulp fiction movie poster"),
    ("Inception", "2010", "inception movie poster"),
    ("Fight Club", "1999", "fight club movie poster"),
    ("Forrest Gump", "1994", "forrest gump movie poster"),
    ("The Matrix", "1999", "matrix movie poster"),
    ("Interstellar", "2014", "interstellar space movie"),
    ("The Lord of the Rings: The Fellowship of the Ring", "2001", "lord of the rings fellowship poster"),
    ("The Lord of the Rings: The Two Towers", "2002", "lord of the rings two towers poster"),
    ("The Lord of the Rings: The Return of the King", "2003", "lord of the rings return of the king poster"),
    ("Gladiator", "2000", "gladiator movie poster"),
    ("Se7en", "1995", "seven movie poster"),
    ("La La Land", "2016", "la la land movie poster"),
    ("Parasite", "2019", "parasite korean movie poster"),
    ("Whiplash", "2014", "whiplash movie poster"),
    ("The Silence of the Lambs", "1991", "silence of the lambs poster"),
    ("Goodfellas", "1990", "goodfellas movie poster"),
    ("City of God", "2002", "city of god movie poster"),
]



def load_json_safe(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.exists():
        print(f"⚠️ Fichier {path} introuvable, je skippe.")
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def seed_items_and_synthetic():
    db = SessionLocal()
    try:
        if db.query(Item).count() == 0:
            print("Seeding items from constants + JSON…")

            # 1) Couleurs (toujours en dur)
            for name, hex_code in COLOR_ITEMS:
                db.add(Item(
                    mode="colors",
                    name=name,
                    hex=hex_code,
                ))

            # 2) Politiciens depuis politicians.json
            pol_data = load_json_safe("politicians.json")
            for row in pol_data:
                db.add(Item(
                    mode="politicians",
                    name=row["name"],
                    subtitle=row.get("subtitle") or "",
                    image_keyword=row.get("image_keyword") or row["name"] + " portrait",
                    image_url=row.get("image_url"),
                ))

            # 3) Destinations depuis destinations.json (50 FR + 50 monde)
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

            # 4) Films depuis movies.json
            movies_data = load_json_safe("movies.json")
            for row in movies_data:
                # On peut construire une vraie URL d’affiche TMDb à partir de poster_path
                # base recommandée : https://image.tmdb.org/t/p/w500 + poster_path
                poster_path = row.get("poster_path")
                if poster_path:
                    tmdb_image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                else:
                    tmdb_image_url = None

                db.add(Item(
                    mode="movies",
                    name=row["name"],
                    subtitle=row.get("subtitle") or "",  # ici l’année
                    image_keyword=row.get("image_keyword") or (row["name"] + " movie poster"),
                    image_url=tmdb_image_url,
                ))

            db.commit()

        # Seed synthétique des ratings (inchangé, mais par mode)
        synthetic_count = db.query(Rating).filter(Rating.is_synthetic == True).count()
        if synthetic_count == 0:
            print("Seeding synthetic ratings…")
            for mode in MODES.keys():
                items = db.query(Item).filter(Item.mode == mode).all()
                item_ids = [i.id for i in items]
                if not item_ids:
                    continue
                num_users = 100
                for u in range(1, num_users + 1):
                    user_id = f"synthetic_{mode}_{u}"
                    for iid in random.sample(item_ids, min(3, len(item_ids))):
                        db.add(
                            Rating(
                                user_id=user_id,
                                mode=mode,
                                item_id=iid,
                                rating=random.randint(2, 5),
                                is_synthetic=True,
                            )
                        )
            db.commit()
    finally:
        db.close()



seed_items_and_synthetic()

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
        real_users = {r.user_id for r in real_ratings}
        num_real_users = len(real_users)

        synthetic_ratings = []
        if num_real_users < 100:
            synthetic_ratings = db.query(Rating).filter(
                Rating.is_synthetic == True,
                Rating.mode == mode,
            ).all()

        rows = []
        for r in real_ratings + synthetic_ratings:
            rows.append({"user": r.user_id, "item": r.item_id, "rating": r.rating})

        if len(rows) < 10:
            # tout début : on prend que le synthétique
            synthetic_ratings = db.query(Rating).filter(
                Rating.is_synthetic == True,
                Rating.mode == mode,
            ).all()
            rows = [{"user": r.user_id, "item": r.item_id, "rating": r.rating}
                    for r in synthetic_ratings]

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


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(title="Multi-mode Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # simple pour un side-project
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Multi-mode recommender backend running"}


@app.get("/modes")
def get_modes():
    """Renvoie les modes disponibles."""
    return [{"id": k, "label": v} for k, v in MODES.items()]


@app.get("/new-session", response_model=str)
def new_session():
    return str(uuid.uuid4())


@app.get("/sample-items", response_model=List[ItemOut])
def sample_items(mode: str = Query("colors")):
    if mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")
    db = SessionLocal()
    try:
        items = db.query(Item).filter(Item.mode == mode).all()
        if len(items) == 0:
            raise HTTPException(status_code=404, detail="No items for this mode")
        sample_size = min(3, len(items))
        return random.sample(items, sample_size)
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


@app.post("/train-and-predict", response_model=TrainResult)
def train_and_predict(user_id: str, mode: str):
    if mode not in MODES:
        raise HTTPException(status_code=400, detail="Unknown mode")

    db = SessionLocal()
    try:
        data = build_surprise_dataset(mode)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        algorithms = [
            ("SVD", SVD()),
            ("KNNBaseline", KNNBaseline()),
            ("NMF", NMF()),
        ]

        leaderboard: List[Dict] = []
        algo_predictions = {}

        for name, algo in algorithms:
            algo.fit(trainset)
            preds = algo.test(testset)
            rmse = accuracy.rmse(preds, verbose=False)
            leaderboard.append({"model_name": name, "rmse": rmse})
            algo_predictions[name] = {"algo": algo, "preds": preds}

        leaderboard.sort(key=lambda x: x["rmse"])
        for idx, entry in enumerate(leaderboard, start=1):
            entry["rank"] = idx

        best_model_name = leaderboard[0]["model_name"]
        best_model_rmse = leaderboard[0]["rmse"]
        best_algo = algo_predictions[best_model_name]["algo"]
        best_preds_test = algo_predictions[best_model_name]["preds"]
        confusion_matrix = compute_confusion_matrix(best_preds_test)

        # prédictions pour l'utilisateur sur tous les items de ce mode
        all_items = db.query(Item).filter(Item.mode == mode).all()
        rated_ids = {
            r.item_id
            for r in db.query(Rating).filter(
                Rating.user_id == user_id,
                Rating.mode == mode,
                Rating.is_synthetic == False,
            ).all()
        }
        remaining_items = [i for i in all_items if i.id not in rated_ids]

        predictions = []
        for it in remaining_items:
            est = best_algo.predict(uid=user_id, iid=it.id).est
            predictions.append(
                {
                    "item_id": it.id,
                    "name": it.name,
                    "subtitle": it.subtitle,
                    "hex": it.hex,
                    "image_keyword": it.image_keyword,
                    "image_url": it.image_url,
                    "predicted_rating": float(round(est, 2)),
                }
            )

        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)

        return TrainResult(
            predictions=[PredictionOut(**p) for p in predictions],
            leaderboard=[LeaderboardEntry(**e) for e in leaderboard],
            best_model_name=best_model_name,
            best_model_rmse=best_model_rmse,
            confusion_matrix=confusion_matrix,
        )
    finally:
        db.close()
