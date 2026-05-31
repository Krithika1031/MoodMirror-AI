from flask_cors import CORS
from flask import Flask, request, jsonify
from deepface import DeepFace
import base64

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json["image"]

    image_data = data.split(",")[1]

    with open("capture.jpg", "wb") as f:
        f.write(base64.b64decode(image_data))

    result = DeepFace.analyze(
        img_path="capture.jpg",
        actions=["emotion"],
        enforce_detection=False
    )

    emotion = result[0]["emotion"]

    return jsonify({
        "happy": round(float(emotion["happy"]), 2),
        "neutral": round(float(emotion["neutral"]), 2),
        "sad": round(float(emotion["sad"]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)