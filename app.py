from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model import predict_pd  
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "audio" not in request.files:
        return jsonify({"error": "Both image and audio files are required!"}), 400

    image_file = request.files["image"]
    audio_file = request.files["audio"]

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)

    image_file.save(image_path)
    audio_file.save(audio_path)

    try:
        prediction_result = predict_pd(image_path, audio_path)

        if prediction_result is None:
            return jsonify({"error": "Prediction failed!"}), 500

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_prediction": bool(prediction_result["image_prediction"]) if prediction_result["image_prediction"] is not None else None,
            "image_confidence": float(prediction_result["image_confidence"]) if prediction_result["image_confidence"] is not None else None,
            "voice_prediction": bool(prediction_result["voice_prediction"]) if prediction_result["voice_prediction"] is not None else None,
            "voice_confidence": float(prediction_result["voice_confidence"]) if prediction_result["voice_confidence"] is not None else None,
            "fused_prediction": bool(prediction_result["fused_prediction"]) if prediction_result["fused_prediction"] is not None else None,
            "fused_confidence": float(prediction_result["fused_confidence"]) if prediction_result["fused_confidence"] is not None else None,
            "severity_score": float(prediction_result["severity_score"]) if prediction_result["severity_score"] is not None else None
        }


        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Server error during prediction."}), 500

if __name__ == "__main__":
    app.run(debug=True)
