"""Simple Flask interface for MRI image prediction demo."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from src.config import MODEL_PATH

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "bmp"}


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the upload form and prediction result page."""
    prediction_text = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            upload_dir = Path(app.config["UPLOAD_FOLDER"])
            upload_dir.mkdir(exist_ok=True)
            image_path = upload_dir / file.filename
            file.save(image_path)

            if not MODEL_PATH.exists():
                prediction_text = "The trained model is not available yet. Please train the model first."
            else:
                prediction_text = "AI research prototype prediction is ready once the model is trained and loaded."
                confidence = None

    return render_template("index.html", prediction=prediction_text, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
