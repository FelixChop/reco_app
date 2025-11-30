#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour construire automatiquement les datasets :
- 50 personnalités politiques françaises (Wikidata)
- 1000 destinations de vacances, avec un panel varié (Wikidata)
- 100 films (TMDb API)
- 100 musiques (MusicBrainz API)

Résultats : fichiers JSON dans le dossier ./data
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # à définir dans ton env
MUSICBRAINZ_USER_AGENT = (
    "color-reco-app/1.0 (contact: ton_email@example.com)"
)  # à personnaliser

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


def _sparql_request(query: str) -> Dict:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": MUSICBRAINZ_USER_AGENT,
    }
    endpoint = "https://query.wikidata.org/sparql"
    r = requests.get(endpoint, params={"query": query}, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def _extract_id(entity_uri: str) -> str:
    return entity_uri.rsplit("/", 1)[-1]


def _row_value(row: Dict, key: str) -> Optional[str]:
    return row.get(key, {}).get("value")


def fetch_current_french_parties(limit: int = 12) -> List[Tuple[str, str]]:
    """Retourne les partis politiques français actuels classés par popularité (sitelinks)."""

    sparql = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>

    SELECT ?party ?partyLabel ?sitelinks WHERE {{
      ?party wdt:P31 wd:Q7278 ;      # parti politique
             wdt:P17 wd:Q142 .      # pays : France

      FILTER(NOT EXISTS {{ ?party wdt:P576 ?dissolved . }})
      ?party wikibase:sitelinks ?sitelinks .

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "fr,en" .
      }}
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT {limit}
    """

    data = _sparql_request(sparql)
    parties: List[Tuple[str, str]] = []
    for row in data.get("results", {}).get("bindings", []):
        uri = _row_value(row, "party")
        label = _row_value(row, "partyLabel")
        if not uri or not label:
            continue
        parties.append((_extract_id(uri), label))
    return parties


def fetch_party_members(party_id: str, party_label: str, limit: int = 8) -> List[Dict]:
    """Récupère des personnalités françaises liées à un parti donné."""

    sparql = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?person ?personLabel ?image ?dob ?sitelinks WHERE {{
      VALUES ?party {{ wd:{party_id} }}

      ?person wdt:P31 wd:Q5 ;
              wdt:P106 wd:Q82955 ;
              wdt:P27 wd:Q142 ;
              wdt:P102 ?party ;
              wdt:P569 ?dob .

      FILTER(?dob >= "1950-01-01T00:00:00Z"^^xsd:dateTime)
      FILTER(NOT EXISTS {{ ?person wdt:P570 ?dod . }})

      OPTIONAL {{ ?person wdt:P18 ?image . }}
      ?person wikibase:sitelinks ?sitelinks .

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "fr,en" .
      }}
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT {limit}
    """

    data = _sparql_request(sparql)
    members: List[Dict] = []
    for row in data.get("results", {}).get("bindings", []):
        name = _row_value(row, "personLabel")
        image_url = _row_value(row, "image")
        if not name:
            continue
        members.append(
            {
                "name": name,
                "subtitle": party_label,
                "image_url": image_url,
                "image_keyword": name + " portrait",
            }
        )
    return members


def fetch_french_politicians(limit: int = 50) -> List[Dict]:
    """
    Politiciens français actuels, issus des principaux partis politiques français,
    avec un focus sur les personnalités les plus médiatisées (sitelinks).
    """

    parties = fetch_current_french_parties()
    aggregated: Dict[str, Dict] = {}  # name -> entry

    for party_id, party_label in parties:
        for member in fetch_party_members(party_id, party_label, limit=8):
            name = member["name"]
            if name in aggregated:
                # privilégier une entrée avec image ou sous-titre renseigné
                if not aggregated[name].get("image_url") and member.get("image_url"):
                    aggregated[name] = member
                continue
            aggregated[name] = member
        if len(aggregated) >= limit:
            break

    # S'assurer qu'on a au plus `limit` entrées
    return list(aggregated.values())[:limit]




# --------------------------------------------------------------------
# 2. Destinations de vacances (Wikidata)
# --------------------------------------------------------------------


def _normalize_destination(
    name: str,
    country: str,
    subtitle: Optional[str],
    image_url: Optional[str],
    is_french: bool,
) -> Dict:
    return {
        "name": name,
        "subtitle": subtitle or country,
        "country": country,
        "image_url": image_url,
        "image_keyword": f"{name} {country} travel",
        "is_french": is_french,
    }


def _collect_destinations_from_rows(
    rows: Iterable[Dict],
    default_country: str,
    subtitle_key: str = "regionLabel",
    country_key: str = "countryLabel",
) -> List[Dict]:
    destinations: List[Dict] = []
    for row in rows:
        name = _row_value(row, "placeLabel") or _row_value(row, "cityLabel") or _row_value(row, "destinationLabel")
        if not name:
            continue
        image_url = _row_value(row, "image")
        country = _row_value(row, country_key) or default_country
        subtitle = _row_value(row, subtitle_key)
        destinations.append(
            _normalize_destination(
                name=name,
                country=country,
                subtitle=subtitle,
                image_url=image_url,
                is_french=country.lower() == "france",
            )
        )
    return destinations


def _query_destinations(query: str) -> List[Dict]:
    data = _sparql_request(query)
    return data.get("results", {}).get("bindings", [])


def fetch_diverse_destinations(target_count: int = 1000) -> List[Dict]:
    """
    Construit un large panel (~1000) de destinations :
    - Plus beaux villages de France (association officielle)
    - Grandes villes touristiques françaises
    - Destinations dans les pays limitrophes
    - Destinations internationales populaires
    """

    destinations: List[Dict] = []
    seen: set[Tuple[str, str]] = set()

    # 1) Plus Beaux Villages de France
    villages_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>

    SELECT ?place ?placeLabel ?image ?regionLabel WHERE {
      ?place wdt:P463 wd:Q1513319 ;      # membre de l'association
             wdt:P17 wd:Q142 .           # France
      OPTIONAL { ?place wdt:P131 ?region . }
      OPTIONAL { ?place wdt:P18 ?image . }
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "fr,en" .
      }
    }
    """
    villages_rows = _query_destinations(villages_query)
    for item in _collect_destinations_from_rows(villages_rows, default_country="France"):
        key = (item["name"], item["country"])
        if key in seen:
            continue
        seen.add(key)
        destinations.append(item)

    # 2) Grandes villes françaises (population > 50k)
    french_cities_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>

    SELECT ?city ?cityLabel ?image ?regionLabel WHERE {
      ?city wdt:P31/wdt:P279* wd:Q484170 ;  # lieu habité
            wdt:P17 wd:Q142 ;               # France
            wdt:P1082 ?population .
      FILTER(?population >= 50000)
      OPTIONAL { ?city wdt:P131 ?region . }
      OPTIONAL { ?city wdt:P18 ?image . }
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "fr,en" .
      }
    }
    ORDER BY DESC(?population)
    LIMIT 300
    """
    french_rows = _query_destinations(french_cities_query)
    for item in _collect_destinations_from_rows(french_rows, default_country="France", subtitle_key="regionLabel", country_key="countryLabel"):
        key = (item["name"], item["country"])
        if key in seen:
            continue
        seen.add(key)
        destinations.append(item)

    # 3) Pays limitrophes (villes > 100k)
    neighboring_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>

    SELECT ?city ?cityLabel ?image ?countryLabel WHERE {
      VALUES ?country { wd:Q31 wd:Q40 wd:Q39 wd:Q38 wd:Q142 wd:Q183 wd:Q145 wd:Q29 wd:Q228 wd:Q23666 }
      ?city wdt:P31/wdt:P279* wd:Q515 ;
            wdt:P17 ?country ;
            wdt:P1082 ?population .
      FILTER(?population >= 100000)
      OPTIONAL { ?city wdt:P18 ?image . }
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "fr,en" .
      }
    }
    ORDER BY DESC(?population)
    LIMIT 300
    """
    neighboring_rows = _query_destinations(neighboring_query)
    for item in _collect_destinations_from_rows(neighboring_rows, default_country="", subtitle_key="countryLabel", country_key="countryLabel"):
        key = (item["name"], item["country"])
        if key in seen:
            continue
        seen.add(key)
        destinations.append(item)

    # 4) Destinations internationales populaires (par nombre de sitelinks)
    global_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>

    SELECT ?destination ?destinationLabel ?countryLabel ?image ?sitelinks WHERE {
      ?destination wdt:P31/wdt:P279* wd:Q515 ;
                   wdt:P17 ?country ;
                   wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 30)
      OPTIONAL { ?destination wdt:P18 ?image . }
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "fr,en" .
      }
    }
    ORDER BY DESC(?sitelinks)
    LIMIT 500
    """
    global_rows = _query_destinations(global_query)
    for item in _collect_destinations_from_rows(global_rows, default_country=""):
        key = (item["name"], item["country"])
        if key in seen:
            continue
        seen.add(key)
        destinations.append(item)
        if len(destinations) >= target_count:
            break

    print(
        f"Construit {len(destinations)} destinations (dont {sum(1 for r in destinations if r['is_french'])} françaises)."
    )
    return destinations[:target_count]




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
        fetch_diverse_destinations,
        target_count=1000,
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
