#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour construire automatiquement les datasets :
- 50 personnalités politiques françaises (Wikidata)
- 100 destinations de vacances (Wikipedia: List of cities by international visitors)
- 100 films (TMDb API)
- 100 musiques (MusicBrainz API)

Résultats : fichiers JSON dans le dossier ./data
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict

import requests
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # à définir dans ton env
MUSICBRAINZ_USER_AGENT = (
    "color-reco-app/1.0 (contact: ton_email@example.com)"
)  # à personnaliser

FRENCH_DESTINATIONS = [
    "Paris", "Nice", "Lyon", "Marseille", "Bordeaux",
    "Toulouse", "Strasbourg", "Lille", "Nantes", "Montpellier",
    "Rennes", "Reims", "Dijon", "Grenoble", "Annecy",
    "Aix-en-Provence", "Avignon", "Arles", "Nîmes", "Cannes",
    "Antibes", "Saint-Tropez", "Menton", "Biarritz", "Bayonne",
    "Saint-Jean-de-Luz", "La Rochelle", "Île de Ré", "Mont Saint-Michel", "Saint-Malo",
    "Étretat", "Deauville", "Honfleur", "Chamonix", "Megève",
    "Courchevel", "Val-d'Isère", "Les Arcs", "Carcassonne", "Colmar",
    "Riquewihr", "Eguisheim", "Amiens", "Rouen", "Chartres",
    "Versailles", "Clermont-Ferrand", "Perpignan", "Ajaccio", "Bastia",
]

INTERNATIONAL_DESTINATIONS = [
    ("London", "United Kingdom"),
    ("New York City", "United States"),
    ("Tokyo", "Japan"),
    ("Bangkok", "Thailand"),
    ("Barcelona", "Spain"),
    ("Rome", "Italy"),
    ("Florence", "Italy"),
    ("Venice", "Italy"),
    ("Amsterdam", "Netherlands"),
    ("Berlin", "Germany"),
    ("Prague", "Czech Republic"),
    ("Vienna", "Austria"),
    ("Budapest", "Hungary"),
    ("Lisbon", "Portugal"),
    ("Madrid", "Spain"),
    ("Seville", "Spain"),
    ("Marrakech", "Morocco"),
    ("Dubai", "United Arab Emirates"),
    ("Istanbul", "Turkey"),
    ("Singapore", "Singapore"),
    ("Hong Kong", "China"),
    ("Macau", "China"),
    ("Sydney", "Australia"),
    ("Melbourne", "Australia"),
    ("Auckland", "New Zealand"),
    ("Cape Town", "South Africa"),
    ("Rio de Janeiro", "Brazil"),
    ("Buenos Aires", "Argentina"),
    ("Mexico City", "Mexico"),
    ("Los Angeles", "United States"),
    ("San Francisco", "United States"),
    ("Chicago", "United States"),
    ("Miami", "United States"),
    ("Vancouver", "Canada"),
    ("Toronto", "Canada"),
    ("Montreal", "Canada"),
    ("Bali", "Indonesia"),
    ("Phuket", "Thailand"),
    ("Chiang Mai", "Thailand"),
    ("Seoul", "South Korea"),
    ("Busan", "South Korea"),
    ("Hanoi", "Vietnam"),
    ("Ho Chi Minh City", "Vietnam"),
    ("Siem Reap", "Cambodia"),
    ("Cairo", "Egypt"),
    ("Petra", "Jordan"),
    ("Santorini", "Greece"),
    ("Mykonos", "Greece"),
    ("Reykjavik", "Iceland"),
    ("Cancún", "Mexico"),
]

def fetch_wikipedia_image(title: str, lang: str = "en", thumb_size: int = 600) -> str | None:
    """
    Récupère l'URL de la vignette principale d'une page Wikipédia
    en utilisant l'API PageImages.

    Si aucune image n'est trouvée ou en cas d'erreur, renvoie None.
    """
    endpoint = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": thumb_size,
        "titles": title,
    }

    headers = {
        # User-Agent *très* explicite, comme recommandé par Wikipédia
        # adapte avec ton mail et éventuellement l’URL de ton projet
        "User-Agent": "reco-color-app/1.0 (https://example.com; contact: felix@example.com)",
    }

    try:
        r = requests.get(endpoint, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"⚠️ Erreur lors de l'appel Wikipédia pour '{title}' ({lang}) : {e}")
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    thumb = page.get("thumbnail")
    if not thumb:
        return None

    return thumb.get("source")




# --------------------------------------------------------------------
# 1. Politiciens français (Wikidata SPARQL)
# --------------------------------------------------------------------


def fetch_french_politicians(limit: int = 50) -> List[Dict]:
    """
    Politiciens français actuels (vivants, nés après 1950),
    sans doublons par personne.
    """
    endpoint = "https://query.wikidata.org/sparql"

    sparql = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?person ?personLabel ?partyLabel ?image ?dob
    WHERE {{
      ?person wdt:P31 wd:Q5 ;          # human
              wdt:P106 wd:Q82955 ;     # politician
              wdt:P27 wd:Q142 ;        # French
              wdt:P569 ?dob .          # date of birth

      FILTER(?dob >= "1950-01-01T00:00:00Z"^^xsd:dateTime)
      FILTER(NOT EXISTS {{ ?person wdt:P570 ?dod . }})  # still alive

      OPTIONAL {{ ?person wdt:P102 ?party . }}    # political party
      OPTIONAL {{ ?person wdt:P18 ?image . }}     # image

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "fr,en" .
      }}
    }}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": MUSICBRAINZ_USER_AGENT,
    }

    r = requests.get(endpoint, params={"query": sparql}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    # On dédoublonne côté Python : 1 entrée par name
    by_name = {}  # name -> dict

    for row in data["results"]["bindings"]:
        name = row["personLabel"]["value"]
        party = row.get("partyLabel", {}).get("value")
        image_url = row.get("image", {}).get("value")

        # si on a déjà ce nom, on ne remplace pas (ou on pourrait fusionner)
        if name in by_name:
            # option : si on n'avait pas de parti avant et qu'on en a un maintenant
            if not by_name[name]["subtitle"] and party:
                by_name[name]["subtitle"] = party
            continue

        by_name[name] = {
            "name": name,
            "subtitle": party or "",
            "image_url": image_url,
            "image_keyword": name + " portrait",
        }

    results = list(by_name.values())
    # au cas où on aurait plus de 50, on tronque
    return results[:limit]




# --------------------------------------------------------------------
# 2. Destinations de vacances (Wikipedia top 100 cities by visitors)
# --------------------------------------------------------------------


def fetch_top_destinations(limit: int = 100) -> List[Dict]:
    """
    Construit un dataset de 100 destinations de vacances :
    - 50 destinations françaises (toujours le cas)
    - 50 destinations internationales
    avec une image Wikipédia quand elle est disponible.

    Le `limit` est gardé pour compatibilité mais on cible 100 entrées :
    50 FR + 50 non-FR.
    """
    results: List[Dict] = []

    # 1) Destinations françaises (fr.wikipedia.org)
    for city in FRENCH_DESTINATIONS:
        image_url = fetch_wikipedia_image(city, lang="fr")
        results.append(
            {
                "name": city,
                "subtitle": "France",
                "country": "France",
                "image_url": image_url,
                # utile si ton front veut faire des requêtes Unsplash/Autre
                "image_keyword": f"{city} France travel",
                "is_french": True,
            }
        )
        time.sleep(0.1)  # on évite de spammer l'API

    # 2) Destinations internationales (en.wikipedia.org)
    for city, country in INTERNATIONAL_DESTINATIONS:
        image_url = fetch_wikipedia_image(city, lang="en")
        results.append(
            {
                "name": city,
                "subtitle": country,
                "country": country,
                "image_url": image_url,
                "image_keyword": f"{city} {country} travel",
                "is_french": False,
            }
        )
        time.sleep(0.1)

    # On tronque au cas où, mais normalement on a 100 pile.
    results = results[:limit]

    print(f"Construit {len(results)} destinations (dont {sum(1 for r in results if r['is_french'])} françaises).")
    return results




# --------------------------------------------------------------------
# 3. Films (TMDb API)
# --------------------------------------------------------------------


def fetch_popular_movies(limit: int = 100) -> List[Dict]:
    """
    Utilise TMDb /discover/movie pour récupérer les films les plus populaires.
    Docs: https://developer.themoviedb.org/docs/getting-started 
    Nécessite la variable d'environnement TMDB_API_KEY.
    """
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY n'est pas défini dans les variables d'environnement.")

    base_url = "https://api.themoviedb.org/3/discover/movie"
    headers = {"Accept": "application/json"}
    movies: List[Dict] = []

    page = 1
    while len(movies) < limit:
        params = {
            "api_key": TMDB_API_KEY,
            "sort_by": "popularity.desc",
            "page": page,
            "language": "en-US",
        }
        r = requests.get(base_url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        for m in data.get("results", []):
            title = m.get("title") or m.get("original_title")
            year = (m.get("release_date") or "")[:4]
            poster_path = m.get("poster_path")  # ex: /abc123.jpg
            movies.append(
                {
                    "name": title,
                    "subtitle": year,
                    # pour les images, TMDb recommande d'assembler base_url + size + poster_path 
                    "tmdb_id": m.get("id"),
                    "poster_path": poster_path,
                    "image_keyword": f"{title} movie poster",
                }
            )
            if len(movies) >= limit:
                break

        if page >= data.get("total_pages", page):
            break
        page += 1
        time.sleep(0.25)  # éviter de spammer l'API

    return movies[:limit]


# --------------------------------------------------------------------
# 4. Musiques (MusicBrainz API)
# --------------------------------------------------------------------


def fetch_top_songs(limit: int = 100, tag: str = "pop") -> List[Dict]:
    """
    Utilise l'API MusicBrainz pour récupérer des enregistrements
    associés à un tag (par ex. 'pop', 'rock', etc.).
    Docs: https://musicbrainz.org/doc/MusicBrainz_API 

    On fait quelques requêtes paginées avec 'offset'.
    """
    base_url = "https://musicbrainz.org/ws/2/recording"
    headers = {"User-Agent": MUSICBRAINZ_USER_AGENT}
    songs: List[Dict] = []

    page_size = 25
    offset = 0

    while len(songs) < limit:
        params = {
            "fmt": "json",
            "limit": page_size,
            "offset": offset,
            # requête simple : enregistrements taggés 'pop'
            "query": f'tag:{tag}',
        }
        r = requests.get(base_url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        recs = data.get("recordings", [])
        if not recs:
            break

        for rec in recs:
            title = rec.get("title")
            artists = rec.get("artist-credit") or []
            if artists:
                artist_name = artists[0].get("name")
            else:
                artist_name = "Unknown artist"

            songs.append(
                {
                    "name": title,
                    "subtitle": artist_name,
                    "image_keyword": f"{artist_name} {title} music",
                    "musicbrainz_id": rec.get("id"),
                }
            )
            if len(songs) >= limit:
                break

        offset += page_size
        time.sleep(1)  # MusicBrainz demande de ne pas abuser de l'API

    return songs[:limit]


# --------------------------------------------------------------------
# Utilitaires d'écriture
# --------------------------------------------------------------------


def save_json(path: Path, data: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Écrit {len(data)} éléments dans {path}")

def generate_if_missing(label: str, filename: str, fetch_fn, *args, **kwargs):
    """
    Ne génère le JSON que s'il n'existe pas déjà.
    label    : texte pour les logs
    filename : nom du fichier dans DATA_DIR
    fetch_fn : fonction qui retourne une liste de dicts (ex: fetch_french_politicians)
    *args/**kwargs : paramètres passés à fetch_fn
    """
    path = DATA_DIR / filename
    if path.exists():
        print(f"→ {label}: {filename} existe déjà, je ne le regénère pas.")
        return

    print(f"=== {label} ===")
    data = fetch_fn(*args, **kwargs)
    save_json(path, data)



def main():
    generate_if_missing(
        "Politiciens français",
        "politicians.json",
        fetch_french_politicians,
        limit=50,
    )

    generate_if_missing(
        "Destinations de vacances",
        "destinations.json",
        fetch_top_destinations,
        limit=100,
    )

    generate_if_missing(
        "Films (TMDb)",
        "movies.json",
        fetch_popular_movies,
        limit=100,
    )

    generate_if_missing(
        "Musiques (MusicBrainz ou fallback)",
        "songs.json",
        fetch_top_songs,
        limit=100,
        tag="pop",
    )



if __name__ == "__main__":
    main()
