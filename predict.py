import numpy as np
import pandas as pd
import librosa

# Načítanie modelu zo súboru
def load_model(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        feature_names = lines[0].strip().split()  # Prvý riadok - názvy vlastností
        weights = np.array([float(x) for x in lines[1].split()])
        bias = float(lines[2].strip())
    return feature_names, weights, bias

# Extrahovanie vlastností zo zvukového súboru
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    feature_dict = {
        "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms_mean": np.mean(librosa.feature.rms(y=y)),
        "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "mfcc1_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)[0]),
        "mfcc2_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)[1])
    }
    return feature_dict

# Výpočet min/max hodnôt z datasetu
def compute_min_max(dataset_path):
    df = pd.read_csv(dataset_path)
    feature_names = df.columns[:-1].tolist()  # Extrahovanie názvov vlastností 
    min_vals = df.iloc[:, :-1].min().values
    max_vals = df.iloc[:, :-1].max().values
    return feature_names, min_vals, max_vals

# Min-Max škálovanie vlastností
def min_max_scale(features, feature_names, min_vals, max_vals, dataset_feature_names):
    # Vytvorenie mapovania názvov vlastností na ich indexy v datasete
    feature_map = {name: i for i, name in enumerate(dataset_feature_names)}
    
    # Transformácia na správne poradie
    scaled_features = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in feature_map:
            value = features[name]
            min_val = min_vals[feature_map[name]]
            max_val = max_vals[feature_map[name]]
            scaled_features[i] = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0
        else:
            print(f"Warning: Feature {name} missing from dataset. Assigning zero.")
            scaled_features[i] = 0.0  # Priradenie nuly, ak vlastnosť chýba
    
    return scaled_features

# Predikcia žánru zo zvukového súboru
def predict_genre(audio_file, model_path, dataset_path):
    # Načítanie min/max hodnôt a názvov vlastností z datasetu
    dataset_feature_names, min_vals, max_vals = compute_min_max(dataset_path)
    # Extrahovanie vlastností zo zvuku
    extracted_features = extract_features(audio_file)
    # Načítanie modelu
    model_feature_names, weights, bias = load_model(model_path)
    # Škálovanie vlastností a ich zoradenie podľa modelu
    scaled_features = min_max_scale(extracted_features, model_feature_names, min_vals, max_vals, dataset_feature_names)
    # Výpočet predikcie
    prediction = np.dot(scaled_features, weights) - bias
    return "Hip-Hop" if prediction >= 0 else "Classical"

