#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour construire automatiquement les datasets :
- 100 personnalités politiques françaises (Wikidata)
- 1000 destinations de vacances, avec un panel varié (Wikidata)
- 1000 films (TMDb API)
- 1000 musiques (MusicBrainz API)

Résultats : fichiers JSON dans le dossier ./data
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

MUSICBRAINZ_USER_AGENT = (
    "ColorRecoApp/1.0 ( https://github.com/felix/reco_app; contact: felixrevert@gmail.com )"
)

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
    """Récupère des personnalités françaises liées à un parti donné, avec leur fonction actuelle."""

    sparql = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?person ?personLabel ?image ?dob ?sitelinks (GROUP_CONCAT(DISTINCT ?jobLabel; separator=", ") AS ?jobs) WHERE {{
      VALUES ?party {{ wd:{party_id} }}

      ?person wdt:P31 wd:Q5 ;
              wdt:P106 wd:Q82955 ;
              wdt:P27 wd:Q142 ;
              wdt:P569 ?dob .
      
      # IMPORTANT : On vérifie l'appartenance au parti via le statement P102
      # et on s'assure qu'il n'y a PAS de date de fin (P582)
      ?person p:P102 ?partyStmt .
      ?partyStmt ps:P102 ?party .
      FILTER(NOT EXISTS {{ ?partyStmt pq:P582 ?endParty }}) .

      FILTER(?dob >= "1950-01-01T00:00:00Z"^^xsd:dateTime)
      FILTER(NOT EXISTS {{ ?person wdt:P570 ?dod . }})

      OPTIONAL {{ ?person wdt:P18 ?image . }}
      ?person wikibase:sitelinks ?sitelinks .
      
      # Récupérer la fonction actuelle (P39 sans date de fin P582)
      OPTIONAL {{
        ?person p:P39 ?stmt .
        ?stmt ps:P39 ?job .
        FILTER(NOT EXISTS {{ ?stmt pq:P582 ?end }}) .
        ?job rdfs:label ?jobLabel .
        FILTER(LANG(?jobLabel) = "fr")
      }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "fr,en" .
      }}
    }}
    GROUP BY ?person ?personLabel ?image ?dob ?sitelinks
    ORDER BY DESC(?sitelinks)
    LIMIT {limit}
    """

    data = _sparql_request(sparql)
    members: List[Dict] = []
    for row in data.get("results", {}).get("bindings", []):
        name = _row_value(row, "personLabel")
        image_url = _row_value(row, "image")
        jobs = _row_value(row, "jobs")
        
        if not name:
            continue
            
        # Nettoyage des jobs
        job_display = ""
        if jobs:
            first_job = jobs.split(", ")[0] 
            job_display = first_job.capitalize()
            
        subtitle = f"{job_display} ({party_label})" if job_display else party_label
        
        # Plus de correctifs manuels, on fait confiance à la requête stricte

        members.append(
            {
                "name": name,
                "subtitle": subtitle,
                "image_url": image_url,
                "image_keyword": name + " portrait",
            }
        )
    return members


def fetch_french_politicians(limit: int = 100) -> List[Dict]:
    """
    Politiciens français actuels, issus des principaux partis politiques français.
    On force la récupération de certains partis clés pour éviter les erreurs (ex: Macron -> PS).
    """
    
    # Liste de partis prioritaires avec leurs IDs Wikidata pour s'assurer qu'on tape juste
    # Q22668282 = La République En Marche / Renaissance
    # Q19509 = Rassemblement National
    # Q217706 = La France Insoumise
    # Q133125 = Les Républicains
    # Q3052772 = Emmanuel Macron (si on veut le cibler, mais ici on cible les partis)
    # Q7278 = Parti politique (générique)
    
    # On va faire une requête un peu différente : on itère sur une liste de partis "cibles"
    # pour être sûr d'avoir les bonnes étiquettes.
    
    target_parties = [
        ("Q22668282", "Renaissance"),
        ("Q205150", "Rassemblement National"),
        ("Q27978402", "La France Insoumise"),
        ("Q20012759", "Les Républicains"),
        ("Q170972", "Parti Socialiste"),
        ("Q591385", "Europe Écologie Les Verts"),
        ("Q108846587", "Horizons"),
        ("Q192821", "Parti Communiste Français"),
    ]
    
    aggregated: Dict[str, Dict] = {}
    
    # On essaie d'équilibrer un peu : limit / nb_partis
    per_party_limit = max(10, int(limit / len(target_parties)) + 5)

    for party_id, party_label in target_parties:
        members = fetch_party_members(party_id, party_label, limit=per_party_limit)
        for member in members:
            name = member["name"]
            # Si déjà présent, on met à jour si la nouvelle entrée a une image et pas l'ancienne
            if name in aggregated:
                if not aggregated[name].get("image_url") and member.get("image_url"):
                    aggregated[name] = member
                # Si Macron est trouvé dans Renaissance, on écrase l'entrée PS éventuelle
                if name == "Emmanuel Macron" and party_label == "Renaissance":
                     aggregated[name] = member
                continue
            aggregated[name] = member
            
    # Si on n'a pas assez, on complète avec la méthode générique
    if len(aggregated) < limit:
        generic_parties = fetch_current_french_parties(limit=10)
        for pid, plabel in generic_parties:
            # On skip ceux qu'on a déjà traités explicitement
            if pid in [t[0] for t in target_parties]:
                continue
                
            members = fetch_party_members(pid, plabel, limit=5)
            for m in members:
                if m["name"] not in aggregated:
                    aggregated[m["name"]] = m
            
            if len(aggregated) >= limit:
                break

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


def fetch_popular_movies(limit: int = 1000) -> List[Dict]:
    """
    Utilise TMDb /discover/movie pour récupérer les films les plus populaires en France
    sur les 30 dernières années.
    """
    if not TMDB_API_KEY:
        print("⚠️ TMDB_API_KEY manquant pour les films.")
        return []

    base_url = "https://api.themoviedb.org/3/discover/movie"
    headers = {"Accept": "application/json"}
    movies: List[Dict] = []
    seen_ids = set()

    # On boucle sur les années pour avoir de la variété et atteindre 1000 films
    # 1000 films / 30 ans ~= 33 films par an.
    start_year = 1994
    end_year = 2024
    
    # On va faire une requête par année pour être sûr de bien couvrir la période
    # On trie par revenue pour avoir les "box office" ou popularity
    
    for year in range(end_year, start_year - 1, -1):
        if len(movies) >= limit:
            break
            
        page = 1
        # On récupère ~40 films par an pour avoir de la marge
        movies_per_year = 0
        target_per_year = 35 
        
        while movies_per_year < target_per_year:
            params = {
                "api_key": TMDB_API_KEY,
                "sort_by": "popularity.desc", # ou revenue.desc
                "page": page,
                "language": "fr-FR", # Titres en français
                "region": "FR",      # Sorties France
                "primary_release_year": year,
                "vote_count.gte": 100 # Filtrer les trucs obscurs
            }
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json()
                
                results = data.get("results", [])
                if not results:
                    break
                    
                for m in results:
                    tmdb_id = m.get("id")
                    if tmdb_id in seen_ids:
                        continue
                        
                    title = m.get("title")
                    release_date = m.get("release_date") or ""
                    poster_path = m.get("poster_path")
                    
                    if not poster_path:
                        continue

                    movies.append({
                        "name": title,
                        "subtitle": release_date[:4],
                        "tmdb_id": tmdb_id,
                        "poster_path": poster_path,
                        "image_keyword": f"{title} movie poster",
                    })
                    seen_ids.add(tmdb_id)
                    movies_per_year += 1
                    
                    if len(movies) >= limit:
                        break
                
                if page >= data.get("total_pages", page):
                    break
                page += 1
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching movies for {year}: {e}")
                break
                
        print(f"  -> {year}: {movies_per_year} films récupérés.")

    return movies[:limit]


# --------------------------------------------------------------------
# 4. Musiques (Spotify API)
# --------------------------------------------------------------------


def get_spotify_token() -> str | None:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    
    auth_url = "https://accounts.spotify.com/api/token"
    try:
        r = requests.post(
            auth_url,
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print(f"❌ Erreur auth Spotify: {e}")
        return None

def fetch_spotify_songs(limit: int = 1000, query: str = "") -> List[Dict]:
    """
    Récupère des musiques via l'API Spotify.
    On itère sur les années pour avoir un top 1000 sur 30 ans.
    """
    token = get_spotify_token()
    if not token:
        print("⚠️ Pas de credentials Spotify, impossible de récupérer les musiques.")
        return []

    headers = {"Authorization": f"Bearer {token}"}
    search_url = "https://api.spotify.com/v1/search"
    songs: List[Dict] = []
    seen_ids = set()
    
    start_year = 1994
    end_year = 2024
    
    # 1000 songs / 30 years ~= 33 songs/year
    target_per_year = 35
    
    for year in range(end_year, start_year - 1, -1):
        if len(songs) >= limit:
            break
            
        q = f"year:{year} genre:pop" # On peut varier les genres si besoin, ou juste year:{year}
        
        offset = 0
        batch_size = 50
        year_songs = 0
        
        while year_songs < target_per_year:
            params = {
                "q": q,
                "type": "track",
                "limit": batch_size,
                "offset": offset,
                "market": "FR" # Focus marché français
            }
            try:
                r = requests.get(search_url, headers=headers, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                tracks = data.get("tracks", {}).get("items", [])
                
                if not tracks:
                    break
                    
                for t in tracks:
                    if year_songs >= target_per_year:
                        break
                        
                    sid = t.get("id")
                    if not sid or sid in seen_ids:
                        continue
                    
                    # Check preview or popularity?
                    # On prend tout ce qui a un ID valide.
                        
                    name = t.get("name")
                    artists = [a["name"] for a in t.get("artists", [])]
                    subtitle = ", ".join(artists)
                    album_images = t.get("album", {}).get("images", [])
                    image_url = album_images[0]["url"] if album_images else None
                    
                    songs.append({
                        "name": name,
                        "subtitle": subtitle,
                        "image_keyword": f"{name} {subtitle} song",
                        "image_url": image_url,
                        "spotify_id": sid
                    })
                    seen_ids.add(sid)
                    year_songs += 1
                    
                    if len(songs) >= limit:
                        break
                
                offset += batch_size
                if offset > 100: # On ne va pas chercher trop loin dans l'année pour garder les tops
                    break
                    
                time.sleep(0.1)

            except Exception as e:
                print(f"❌ Erreur fetch Spotify {year}: {e}")
                break
        
        print(f"  -> {year}: {year_songs} musiques récupérées.")
            
    return songs[:limit]


# --------------------------------------------------------------------
# Utilitaires d'écriture
# --------------------------------------------------------------------


def save_json(path: Path, data: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Écrit {len(data)} éléments dans {path}")

def generate_dataset(label: str, filename: str, fetch_fn, force: bool = False, *args, **kwargs):
    """
    Génère le JSON. Si force=True, écrase l'existant.
    """
    path = DATA_DIR / filename
    if path.exists() and not force:
        print(f"→ {label}: {filename} existe déjà, je ne le regénère pas.")
        return

    print(f"=== {label} ===")
    try:
        data = fetch_fn(*args, **kwargs)
        if data:
            save_json(path, data)
        else:
            print(f"⚠️ Aucune donnée récupérée pour {label}")
    except Exception as e:
        print(f"❌ Erreur pour {label}: {e}")


def main():
    # Force regeneration for politicians to fix "Opposition" labels
    generate_dataset(
        "Politiciens français",
        "politicians.json",
        fetch_french_politicians,
        limit=100,
        force=False
    )

    generate_dataset(
        "Destinations de vacances",
        "destinations.json",
        fetch_diverse_destinations,
        target_count=1000,
        force=False 
    )

    # Movies require API Key
    if TMDB_API_KEY:
        generate_dataset(
            "Films (TMDb)",
            "movies.json",
            fetch_popular_movies,
            limit=1000,
            force=False
        )
    else:
        print("⚠️ TMDB_API_KEY manquant : on saute la génération des films (on garde l'existant si présent).")

    # Songs (Spotify)
    if SPOTIFY_CLIENT_ID:
        generate_dataset(
            "Musiques (Spotify)",
            "songs.json",
            fetch_spotify_songs,
            limit=1000,
            force=False
        )
    else:
        print("⚠️ SPOTIFY_CLIENT_ID manquant : on saute la génération des musiques (on garde l'existant si présent).")
    
    # Optimisation des images
    print("\n=== Optimisation des images ===")
    try:
        import subprocess
        # optimize_images.py est dans le même dossier que ce script (scraping/)
        script_path = Path(__file__).parent / "optimize_images.py"
        subprocess.run(["python3", str(script_path)], check=True)
    except Exception as e:
        print(f"⚠️ Impossible de lancer l'optimisation des images : {e}")

if __name__ == "__main__":
    main()
