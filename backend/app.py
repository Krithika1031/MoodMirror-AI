from flask_cors import CORS
from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

CAPTURES_FOLDER = "captures"
HISTORY_FILE = "history.json"

os.makedirs(CAPTURES_FOLDER, exist_ok=True)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json["image"]
    image_data = data.split(",")[1]

    timestamp = datetime.now()
    filename = timestamp.strftime("scan_%Y%m%d_%H%M%S_%f.jpg")
    image_path = os.path.join(CAPTURES_FOLDER, filename)

    image_bytes = base64.b64decode(image_data)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )

    emotion = result[0]["emotion"]

    scan_data = {
        "image": filename,
        "happy": round(float(emotion["happy"]), 2),
        "neutral": round(float(emotion["neutral"]), 2),
        "sad": round(float(emotion["sad"]), 2),
        "date": timestamp.strftime("%d %B %Y"),
        "time": timestamp.strftime("%I:%M %p")
    }

    history = []

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    history.append(scan_data)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    return jsonify({
        "happy": scan_data["happy"],
        "neutral": scan_data["neutral"],
        "sad": scan_data["sad"]
    })


@app.route("/history", methods=["GET"])
def history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])

    with open(HISTORY_FILE, "r") as f:
        history_data = json.load(f)

    for scan in history_data:
        scan["image"] = "/captures/" + scan["image"]

    return jsonify(history_data)


@app.route("/captures/<filename>")
def get_capture(filename):
    from flask import send_from_directory
    return send_from_directory(CAPTURES_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)