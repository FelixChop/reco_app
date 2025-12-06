# API Documentation

The backend is built with FastAPI and exposes the following endpoints.

## Base URL
`http://localhost:8000` (default)

## Endpoints

### General

#### `GET /`
Returns a health check message.
- **Response:** `{"message": "Multi-mode recommender backend running"}`

#### `GET /modes`
Retrieves the available recommendation modes.
- **Response:** List of objects `{"id": "mode_id", "label": "Mode Label"}`.
- **Example:** `[{"id": "colors", "label": "Colors"}, ...]`

#### `GET /new-session`
Generates a new unique session ID for the user.
- **Response:** A UUID string.
- **Example:** `"550e8400-e29b-41d4-a716-446655440000"`

### Items & Ratings

#### `GET /sample-items`
Fetches a sample of items for the user to rate.
- **Parameters:**
    - `mode` (query, string): The mode to sample items from (default: "colors").
    - `limit` (query, int): Number of items to return (default: 3).
    - `exclude_ids` (query, list[int]): List of item IDs to exclude (already seen/rated).
- **Response:** List of item objects.
- **Example:**
    ```json
    [
        {
            "id": 1,
            "name": "Red",
            "mode": "colors",
            "image_url": "...",
            "image_keyword": "red"
        }
    ]
    ```

#### `POST /rate`
Submits a user rating for an item.
- **Body:**
    ```json
    {
        "user_id": "string",
        "mode": "string",
        "item_id": 123,
        "rating": 5
    }
    ```
- **Response:** `{"status": "ok", "rating_id": 1}`

#### `GET /rating-counts`
Retrieves the number of ratings submitted by a user for each mode.
- **Parameters:**
    - `user_id` (query, string): The user's session ID.
- **Response:** Dictionary mapping mode IDs to rating counts.
- **Example:** `{"colors": 5, "movies": 10}`

### Recommendations

#### `POST /train-and-predict`
Triggers model training and generates recommendations.
- **Parameters:**
    - `user_id` (query, string): The user's session ID.
    - `mode` (query, string): The mode to generate recommendations for.
- **Response:**
    ```json
    {
        "leaderboard": [
            {
                "algorithm": "SVD",
                "rmse": 0.85,
                "rmse_user": 0.5,
                "rank": 1,
                "best_params": {...}
            },
            ...
        ],
        "predictions": [
            {
                "item_id": 45,
                "name": "Blue",
                "est": 4.8,
                "image_url": "..."
            },
            ...
        ]
    }
    ```
