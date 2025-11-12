from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"

SCALER = joblib.load(f"{MODELS_DIR}/backend_scaler.pkl")
KNN = joblib.load(f"{MODELS_DIR}/backend_knn.pkl")
TRACKS = pd.read_pickle(f"{MODELS_DIR}/backend_tracks.pkl")

FEATURE_COLUMNS = [
    "tempo", "zcr", "centroid", "bandwidth", "rolloff",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5"
]

BACKEND_SONGS_API = "https://loginsignup-2.onrender.com/api/songs"

def get_backend_songs():
    try:
        r = requests.get(BACKEND_SONGS_API, timeout=30)
        data = r.json()
        if isinstance(data, dict) and "songs" in data:
            return data["songs"]
        return data
    except Exception as e:
        print("Backend fetch error:", e)
        return []

def cbf_recommend(song_id, backend_songs, n=5):
    seed_rows = TRACKS[TRACKS["id"].astype(str) == str(song_id)]
    if seed_rows.empty:
        return []

    seed_vec = SCALER.transform(seed_rows[FEATURE_COLUMNS])
    dists, idxs = KNN.kneighbors(seed_vec, n_neighbors=min(n + 1, len(TRACKS)))

    ids = TRACKS.iloc[idxs[0]]["id"].astype(str).tolist()
    ids = [i for i in ids if i != str(song_id)]

    meta = {str(s["id"]): s for s in backend_songs}
    return [meta[i] for i in ids if i in meta][:n]

def cf_recommend(user_history, backend_songs, top_n=5):
    if not user_history:
        return []

    df = pd.DataFrame(user_history)
    df = df.groupby("song_id")["plays"].sum().sort_values(ascending=False)
    ids = df.index.astype(str).tolist()

    meta = {str(s["id"]): s for s in backend_songs}
    return [meta[i] for i in ids if i in meta][:top_n]

def hybrid(song_id, user_history, backend_songs):
    cbf = cbf_recommend(song_id, backend_songs)
    cf = cf_recommend(user_history, backend_songs)

    scored = []
    for s in cbf: scored.append((s, 0.6))
    for s in cf:  scored.append((s, 0.4))

    seen = set()
    out = []
    for s, _ in sorted(scored, key=lambda x: x[1], reverse=True):
        sid = str(s["id"])
        if sid not in seen:
            out.append(s)
            seen.add(sid)
        if len(out) >= 5:
            break
    return out

@app.route("/recommend/next", methods=["POST"])
def recommend_next():
    data = request.json
    user_id = str(data.get("user_id"))
    song_id = str(data.get("id"))
    user_history = data.get("history", [])

    backend_songs = get_backend_songs()
    recs = hybrid(song_id, user_history, backend_songs)

    if recs:
        return jsonify({"next_song": recs[0]})

    if user_history:
        top_id = str(sorted(user_history, key=lambda x: x["plays"], reverse=True)[0]["song_id"])
        by_id = {str(s["id"]): s for s in backend_songs}
        if top_id in by_id:
            return jsonify({"next_song": by_id[top_id]})

    return jsonify({"next_song": backend_songs[0]})

@app.route("/recommend/home", methods=["POST"])
def home():
    data = request.json
    user_id = str(data.get("user_id"))
    user_history = data.get("history", [])

    backend_songs = get_backend_songs()

    if not user_history:
        return jsonify({"recommendations": backend_songs[:12]})

    pool = []
    for h in user_history:
        sid = str(h["song_id"])
        pool.extend(hybrid(sid, user_history, backend_songs))

    seen = set()
    final = []
    for s in pool:
        sid = str(s["id"])
        if sid not in seen:
            final.append(s)
            seen.add(sid)
        if len(final) >= 12:
            break

    return jsonify({"recommendations": final})

@app.route("/")
def health():
    return {"status": "ok", "service": "Backend-driven Audio Recommender"}

if __name__ == "__main__":
    app.run(debug=True)