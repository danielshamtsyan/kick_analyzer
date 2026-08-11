"""
app.py -- upload web app for the roundhouse kick analyzer.

This is the "up to the uploading part" layer: a person picks a video file in
the browser, the app saves it, runs the existing pose-analysis pipeline
(pose_extract.py / analyze_kick.py, unmodified), and shows the results --
feedback, confidence score, and the annotated skeleton-overlay video -- back
on a results page.

Run it with:
    python app.py
Then open http://localhost:5000 in a browser.

Deliberately kept to local/single-process Flask for the validation stage --
no auth, no database, no cloud storage. Good enough to test with training
partners on your own machine. Swap in real storage/auth/queueing once the
pipeline itself is validated (see Track 1 / Track 2 in the README) -- no
point building production infrastructure around a pipeline whose accuracy
hasn't been confirmed on real footage yet.
"""

import os
import sys
import uuid

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Make the parent directory's pipeline modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze_kick import run_analysis  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ALLOWED_EXTENSIONS = {"mp4", "mov", "m4v", "avi"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB -- generous for a few seconds of phone video

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return render_template("index.html", error="No file was submitted.")

    file = request.files["video"]
    side = request.form.get("side", "right")

    if file.filename == "":
        return render_template("index.html", error="No file was selected.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error=f"Unsupported file type. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Unique session id so concurrent uploads (e.g. testing with a few training
    # partners back to back) don't overwrite each other's files or results.
    session_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    video_path = os.path.join(UPLOAD_DIR, f"{session_id}.{ext}")
    file.save(video_path)

    out_dir = os.path.join(RESULTS_DIR, session_id)

    try:
        result = run_analysis(video_path, side, out_dir, verbose=False)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Something went wrong processing that video: {exc}",
        )

    return render_template(
        "results.html",
        session_id=session_id,
        side=side,
        result=result,
    )


@app.route("/results/<session_id>/<path:filename>")
def serve_result_file(session_id, filename):
    """Serves the annotated video / confidence CSV for a given session."""
    directory = os.path.join(RESULTS_DIR, session_id)
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
