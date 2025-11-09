from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SCALER = joblib.load(f"{MODELS_DIR}/backend_scaler.pkl")
KNN     = joblib.load(f"{MODELS_DIR}/backend_knn.pkl")
TRACKS  = pd.read_pickle(f"{MODELS_DIR}/backend_tracks.pkl")

FEATURE_COLUMNS = [
    "tempo", "zcr", "centroid", "bandwidth", "rolloff",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5"
]

USER_LISTENS_PATH = f"{DATA_DIR}/user_listens.csv"
if not os.path.exists(USER_LISTENS_PATH):
    pd.DataFrame(columns=["user_id", "song_id", "plays"]).to_csv(USER_LISTENS_PATH, index=False)

BACKEND_SONGS_API = "https://loginsignup-2.onrender.com/api/songs"


def get_backend_songs():
    try:
        r = requests.get(BACKEND_SONGS_API, timeout=60)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "songs" in data:
                return data["songs"]
            return data
    except Exception as e:
        print("Backend fetch error:", e)

    return []


@app.route("/ml/user_event", methods=["POST"])
def save_event():
    data = request.json
    user_id = str(data["user_id"])
    song_id = str(data["song_id"])

    df = pd.read_csv(USER_LISTENS_PATH)
    mask = (df["user_id"] == user_id) & (df["song_id"] == song_id)

    if mask.any():
        df.loc[mask, "plays"] += 1
    else:
        df.loc[len(df)] = [user_id, song_id, 1]

    df.to_csv(USER_LISTENS_PATH, index=False)
    return jsonify({"message": "event_saved"}), 200


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


def cf_recommend(user_id, backend_songs, top_n=5):
    df = pd.read_csv(USER_LISTENS_PATH)
    if df[df["user_id"] == user_id].empty:
        return []

    matrix = df.pivot_table(index="user_id", columns="song_id", values="plays", fill_value=0)
    users = matrix.index.tolist()

    if user_id not in users:
        return []

    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=users, columns=users)

    similar_users = sim_df[user_id].sort_values(ascending=False).iloc[1:6].index.tolist()

    pool = df[df["user_id"].isin(similar_users)]
    top = pool.groupby("song_id")["plays"].sum().sort_values(ascending=False).head(top_n)
    ids = top.index.astype(str).tolist()

    meta = {str(s["id"]): s for s in backend_songs}
    return [meta[i] for i in ids if i in meta]


def hybrid(song_id, user_id, backend_songs):
    cbf = cbf_recommend(song_id, backend_songs)
    cf  = cf_recommend(user_id, backend_songs)

    scored = []
    for r in cbf: scored.append((r, 0.6))
    for r in cf:  scored.append((r, 0.4))

    if not scored:
        return []

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    seen = set()
    out = []
    for r, _ in scored:
        rid = str(r["id"])
        if rid not in seen:
            out.append(r)
            seen.add(rid)
        if len(out) >= 5:
            break

    return out


@app.route("/recommend/next", methods=["POST"])
def recommend_next():
    data = request.json
    user_id = str(data["user_id"])
    song_title = data["song_name"].strip().lower()

    backend_songs = get_backend_songs()
    matches = [s for s in backend_songs if s["title"].strip().lower() == song_title]

    if matches:
        seed = matches[0]
        recs = hybrid(seed["id"], user_id, backend_songs)
        if recs:
            return jsonify({"next_song": recs[0]})

    df = pd.read_csv(USER_LISTENS_PATH)
    user_df = df[df["user_id"] == user_id]
    if not user_df.empty:
        top_id = str(user_df.sort_values("plays", ascending=False).iloc[0]["song_id"])
        by_id = {str(s["id"]): s for s in backend_songs}
        if top_id in by_id:
            return jsonify({"next_song": by_id[top_id]})

    return jsonify({"next_song": backend_songs[0]})


@app.route("/recommend/home", methods=["POST"])
def home():
    data = request.json
    user_id = str(data["user_id"])

    backend_songs = get_backend_songs()

    df = pd.read_csv(USER_LISTENS_PATH)
    if df[df["user_id"] == user_id].empty:
        return jsonify({"recommendations": backend_songs[:12]})

    pool = []
    for _, row in df[df["user_id"] == user_id].iterrows():
        sid = str(row["song_id"])
        pool.extend(hybrid(sid, user_id, backend_songs))

    seen = set()
    final = []
    for r in pool:
        if r["id"] not in seen:
            final.append(r)
            seen.add(r["id"])
        if len(final) >= 12:
            break

    return jsonify({"recommendations": final})


@app.route("/")
def health():
    return {"status": "ok", "service": "Backend Audio Recommender"}


if __name__ == "__main__":
    app.run(debug=True)
