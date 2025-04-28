import joblib
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import nolds
from scipy.stats import entropy
from concurrent.futures import ThreadPoolExecutor

# Load saved model bundles
speech_bundle = joblib.load("speech_deployment_bundle.joblib")
severity_bundle = joblib.load("severity_deployment_bundle.joblib")
resnet_model = load_model("resnet_model.h5")
densenet_model = load_model("densenet_model.h5")

# Unpack models and metadata
scaler_speech = speech_bundle["scaler"]
selector_speech = speech_bundle["selector"]
base_models_speech = speech_bundle["base_models"]
meta_model_speech = speech_bundle["meta_model"]
optimal_threshold_speech = speech_bundle["optimal_threshold"]
selected_features_speech = speech_bundle["selected_features"]
train_medians_speech = speech_bundle["train_medians"]

scaler_severity = severity_bundle["scaler"]
meta_model_severity = severity_bundle["meta_model"]
base_model_severity = severity_bundle["base_models"]
selected_features_severity = severity_bundle["selected_features"]

def calculate_voice_breaks(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=600, sr=sr)
        silent_regions = np.isnan(f0)
        return np.sum(np.diff(silent_regions.astype(int)) > 0)
    except Exception as e:
        print(f"Voice breaks calculation error: {e}")
        return None

def extract_all_audio_features(audio_path, use_percent_scaling=True):
    features = {}
    try:
        y, sr = librosa.load(audio_path, sr=None)
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        pitch_values = pitch.selected_array["frequency"]
        voiced = pitch_values[pitch_values > 0]

        features["Median pitch"] = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
        features["Mean pitch"] = call(pitch, "Get mean", 0, 0, "Hertz")
        features["Minimum pitch"] = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        features["Maximum pitch"] = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        features["Standard deviation"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

        shimmer_local = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([sound, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([sound, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([sound, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        if use_percent_scaling:
            features["Jitter(%)"] = jitter_local * 100
            features["Jitter:RAP"] = jitter_rap * 100
            features["Jitter:PPQ5"] = jitter_ppq5 * 100
            features["Jitter:DDP"] = jitter_ppq5 * 3 * 100

            features["Shimmer"] = shimmer_local * 100
            features["Shimmer:APQ3"] = shimmer_apq3 * 100
            features["Shimmer:APQ5"] = shimmer_apq5 * 100
            features["Shimmer:APQ11"] = shimmer_apq11 * 100
            features["Shimmer:DDA"] = shimmer_apq3 * 3 * 100
        else:
            features["Jitter(%)"] = jitter_local
            features["Jitter:RAP"] = jitter_rap
            features["Jitter:PPQ5"] = jitter_ppq5
            features["Jitter:DDP"] = jitter_ppq5 * 3

            features["Shimmer"] = shimmer_local
            features["Shimmer:APQ3"] = shimmer_apq3
            features["Shimmer:APQ5"] = shimmer_apq5
            features["Shimmer:APQ11"] = shimmer_apq11
            features["Shimmer:DDA"] = shimmer_apq3 * 3

        features["Jitter(Abs)"] = jitter_abs
        features["Shimmer(dB)"] = shimmer_db

        features["Number of pulses"] = call(pp, "Get number of points")
        features["Number of periods"] = call(pp, "Get number of periods", 0, 0, 0.0001, 0.02, 1.3)
        features["Mean period"] = call(pp, "Get mean period", 0, 0, 0.0001, 0.02, 1.3)

        try:
            times = [call(pp, "Get time from index", i + 1) for i in range(call(pp, "Get number of points"))]
            features["Standard deviation of period"] = np.std(np.diff(times)) if len(times) > 1 else None
        except Exception:
            features["Standard deviation of period"] = None

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        features["HTN"] = hnr
        features["HNR"] = hnr
        features["NHR"] = 10 ** (-hnr / 10) if hnr is not None else None

        unvoiced_count = np.sum(np.isnan(pitch_values) | (pitch_values == 0))
        features["Fraction of locally unvoiced frames"] = (unvoiced_count / len(pitch_values)) * 100 if len(pitch_values) > 0 else None

        voice_breaks = calculate_voice_breaks(audio_path)
        features["Degree of voice breaks"] = np.log1p(voice_breaks) * 100 if voice_breaks is not None else None
        features["Number of voice breaks"] = voice_breaks

        if len(voiced) > 0:
            hist, _ = np.histogram(voiced, bins=20, density=True)
            features["PPE"] = entropy(hist + 1e-8)
        else:
            features["PPE"] = None

        y_clean = y[np.abs(y) > 1e-5]
        if len(y_clean) > 0:
            hist, _ = np.histogram(y_clean, bins=30, density=True)
            features["RPDE"] = entropy(hist + 1e-8)
        else:
            features["RPDE"] = None

        try:
            features["DFA"] = nolds.dfa(y)
        except Exception:
            features["DFA"] = None

        features["AC"] = np.mean(np.correlate(y, y, mode="full")) if len(y) > 0 else None

        # Aliases
        features["Jitter (local)"] = features["Jitter(%)"]
        features["Jitter (local, absolute)"] = features["Jitter(Abs)"]
        features["Jitter (rap)"] = features["Jitter:RAP"]
        features["Jitter (ppq5)"] = features["Jitter:PPQ5"]
        features["Jitter (ddp)"] = features["Jitter:DDP"]

        features["Shimmer (local)"] = features["Shimmer"]
        features["Shimmer (local, dB)"] = features["Shimmer(dB)"]
        features["Shimmer (apq3)"] = features["Shimmer:APQ3"]
        features["Shimmer (apq5)"] = features["Shimmer:APQ5"]
        features["Shimmer (apq11)"] = features["Shimmer:APQ11"]
        features["Shimmer (dda)"] = features["Shimmer:DDA"]
        features["NTH"] = features["Number of voice breaks"]

        return pd.DataFrame([features])

    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None


def predict_voice(audio_path):
    print("\nExtracting features for voice prediction...")
    extracted_df = extract_all_audio_features(audio_path, use_percent_scaling=True)

    if extracted_df is None or extracted_df.empty:
        print("Feature extraction failed.")
        return None, None

    scaler_input_features = scaler_speech.feature_names_in_
    full_input = pd.DataFrame(columns=scaler_input_features)
    missing_filled = []

    for col in scaler_input_features:
        if col in extracted_df.columns and pd.notnull(extracted_df[col].iloc[0]):
            full_input.at[0, col] = extracted_df[col].iloc[0]
        else:
            full_input.at[0, col] = train_medians_speech.get(col, 0)
            missing_filled.append(col)

    if missing_filled:
        print(f"Missing features filled with medians: {missing_filled}")

    scaled = scaler_speech.transform(full_input)
    selected = scaled[:, selector_speech.support_]

    meta_input = []
    seen = set()
    for model in base_models_speech:
        model_type = str(type(model))
        if model_type not in seen:
            seen.add(model_type)
            proba = model.predict_proba(selected)[0][1]
            meta_input.append(proba)

    meta_input_df = pd.DataFrame([meta_input], columns=[f"model_{i}" for i in range(len(meta_input))])
    final_proba = meta_model_speech.predict_proba(meta_input_df)[0][1]
    final_prediction = int(final_proba >= optimal_threshold_speech)

    return final_prediction, final_proba


def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        resnet_pred = resnet_model.predict(img_array)[0][0]
        densenet_pred = densenet_model.predict(img_array)[0][0]

        ensemble_prob = (resnet_pred + densenet_pred) / 2.0
        label = int(ensemble_prob >= 0.5)

        return label, float(ensemble_prob)
    
    except Exception as e:
        print(f"Image prediction failed: {e}")
        return None, None


def predict_severity(audio_path):
    print("\nExtracting features for severity score prediction...")
    extracted_df = extract_all_audio_features(audio_path, use_percent_scaling=False)

    if extracted_df is None or extracted_df.empty:
        print("Severity prediction skipped: Feature extraction failed.")
        return None

    try:
        scaler_input_features = scaler_severity.feature_names_in_
        full_input = pd.DataFrame(columns=scaler_input_features)

        missing_filled = []

        for col in scaler_input_features:
            if col in extracted_df.columns and pd.notnull(extracted_df[col].iloc[0]):
                full_input.at[0, col] = extracted_df[col].iloc[0]
            else:
                full_input.at[0, col] = train_medians_speech.get(col, 0)
                missing_filled.append(col)

        if missing_filled:
            print(f"Missing features filled with medians: {missing_filled}")

        scaled_input = scaler_severity.transform(full_input)
        top_features_input = scaled_input[:, selected_features_severity]

        severity_score = meta_model_severity.predict(top_features_input)[0]
        print(f"Predicted Severity Score (UPDRS): {severity_score:.2f}")
        return severity_score

    except Exception as e:
        print(f"Severity prediction failed: {e}")
        return None


def dynamic_fusion(voice_prob, image_prob):
    def confidence(p):
        p = np.clip(p, 0.001, 0.999)
        return 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
    v_weight = confidence(voice_prob)
    i_weight = confidence(image_prob)
    return (v_weight * voice_prob + i_weight * image_prob) / (v_weight + i_weight)

def predict_pd(image_path, audio_path):
    with ThreadPoolExecutor() as executor:
        image_future = executor.submit(predict_image, image_path)
        voice_future = executor.submit(predict_voice, audio_path)

        img_pred, img_prob = image_future.result()
        voice_pred, voice_prob = voice_future.result()

    if voice_prob is not None and img_prob is not None:
        fused_prob = dynamic_fusion(voice_prob, img_prob)
        fused_pred = int(fused_prob >= 0.5)
    else:
        fused_pred = None
        fused_prob = None

    if fused_pred == 1:
        severity_score = predict_severity(audio_path)
        return {
            "image_prediction": bool(img_pred),
            "image_confidence": img_prob,
            "voice_prediction": bool(voice_pred),
            "voice_confidence": voice_prob,
            "fused_prediction": bool(fused_pred),
            "fused_confidence": fused_prob,
            "severity_score": severity_score
        }
    else:
        return {
            "image_prediction": bool(img_pred),
            "image_confidence": img_prob,
            "voice_prediction": bool(voice_pred),
            "voice_confidence": voice_prob,
            "fused_prediction": bool(fused_pred) if fused_pred is not None else None,
            "fused_confidence": fused_prob,
            "severity_score": None
        }
