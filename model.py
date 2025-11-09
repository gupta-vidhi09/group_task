import os
import tempfile
import requests
import numpy as np
import pandas as pd
import joblib
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BACKEND_SONGS_API = "https://loginsignup-2.onrender.com/api/songs"

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "tempo", "zcr", "centroid", "bandwidth", "rolloff",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5"
]


def fetch_backend_songs():
    r = requests.get(BACKEND_SONGS_API, timeout=20)
    r.raise_for_status()

    data = r.json()

    if isinstance(data, dict) and "songs" in data:
        return data["songs"]
    return data


def extract_features(audio_url):

    r = requests.get(audio_url, timeout=30)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(r.content)
        temp_path = tmp.name

    try:
        y, sr = librosa.load(temp_path, sr=22050, mono=True)

        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        zcr = float(librosa.feature.zero_crossing_rate(y=y).mean())
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        mfcc1, mfcc2, mfcc3, mfcc4, mfcc5 = mfcc[:5]

        return {
            "tempo": tempo,
            "zcr": zcr,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "mfcc1": float(mfcc1),
            "mfcc2": float(mfcc2),
            "mfcc3": float(mfcc3),
            "mfcc4": float(mfcc4),
            "mfcc5": float(mfcc5),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



def main():
    print("🔗 Fetching backend songs...")
    songs = fetch_backend_songs()

    rows = []

    for s in songs:
        sid = str(s["id"])
        title = s.get("title", "").strip()
        artist = s.get("artist", "").strip()
        audio_url = s.get("audioUrl")

        try:
            feats = extract_features(audio_url)
            rows.append({
                "id": sid,
                "title": title,
                "artist": artist,
                **feats
            })
            print(f"✅ Extracted: {title}")
        except Exception as e:
            print(f"⚠️ Failed: {title} -> {e}")

    df = pd.DataFrame(rows)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS])

    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(X)

    joblib.dump(scaler, f"{MODELS_DIR}/backend_scaler.pkl")
    joblib.dump(knn, f"{MODELS_DIR}/backend_knn.pkl")
    df.to_pickle(f"{MODELS_DIR}/backend_tracks.pkl")

    print("✅ Training complete!")
    print("Saved:")
    print(" - backend_scaler.pkl")
    print(" - backend_knn.pkl")
    print(" - backend_tracks.pkl")


if __name__ == "__main__":
    main()
