import os
import pandas as pd
import librosa
import numpy as np
import requests
import tempfile
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

BACKEND_SONGS_API = "https://loginsignup-2.onrender.com/api/songs"

def extract_features(audio_url):
    tmp_path = None
    try:
        r = requests.get(audio_url, timeout=30)
        if r.status_code != 200:
            print(f"Failed to fetch: {audio_url}")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=22050, mono=True)

        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        mfcc_means = [np.mean(mfcc[i]) for i in range(5)]

        features = {
            "tempo": tempo,
            "zcr": zcr,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "mfcc1": mfcc_means[0],
            "mfcc2": mfcc_means[1],
            "mfcc3": mfcc_means[2],
            "mfcc4": mfcc_means[3],
            "mfcc5": mfcc_means[4],
        }
        return features

    except Exception as e:
        print(f"Error processing {audio_url}: {e}")
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def fetch_backend_songs():
    print("Fetching backend songs...")
    try:
        res = requests.get(BACKEND_SONGS_API, timeout=60)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "songs" in data:
                return data["songs"]
            return data
    except Exception as e:
        print("Error fetching songs:", e)
    return []

def main():
    songs = fetch_backend_songs()
    if not songs:
        print("No songs fetched from backend.")
        return

    print(f"{len(songs)} songs fetched. Extracting features...")

    all_features = []
    for s in songs:
        if "audioUrl" not in s or not s["audioUrl"]:
            continue

        features = extract_features(s["audioUrl"])
        if features:
            features["id"] = s["id"]
            features["title"] = s["title"]
            features["artist"] = s["artist"]
            all_features.append(features)

    df = pd.DataFrame(all_features)
    print(f"Features extracted for {len(df)} songs.")

    feature_cols = [
        "tempo", "zcr", "centroid", "bandwidth", "rolloff",
        "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5"
    ]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(scaled)

    joblib.dump(scaler, f"{MODELS_DIR}/backend_scaler.pkl")
    joblib.dump(model, f"{MODELS_DIR}/backend_knn.pkl")
    df.to_pickle(f"{MODELS_DIR}/backend_tracks.pkl")

    print("Training complete. Model files saved in /models.")

if __name__ == "__main__":
    main()
