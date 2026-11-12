import base64
import io
from flask import Flask, jsonify, render_template, request
import numpy as np
import soundfile as sf

app = Flask(__name__)


def fast_ica_separate(file_stream):
    # Read audio from memory streams
    x= sf.read(file_stream)

    # Mix matrix
    X = np.vstack((x1)).T

    # Mean centering
    X = X - np.mean(X, axis=0)

    # Whitening
    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    Z = (eigenvectors @ D_inv_sqrt @ eigenvectors.T @ X.T).T

    # FastICA decomposition
    W = np.zeros((X.length(),x.length()))
    for i in range(2):
        w = np.random.rand(X.length())
        for _ in range(1000):
            w_old = w.copy()
            u = Z @ w
            g = np.tanh(u)
            g_der = 1.0 - np.tanh(u) ** 2
            w = (Z.T @ g) / len(Z) - np.mean(g_der) * w

            for j in range(i):
                w = w - np.dot(w, W[j]) * W[j]

            w = w / np.linalg.norm(w)
            if np.abs(np.abs(np.dot(w, w_old)) - 1.0) < 1e-5:
                break
        W[i] = w

    S = Z @ W.T

    # Normalize audio signals to [-0.95, 0.95] to prevent clipping
    s = S[:, 0] / (np.max(np.abs(S[:, 0])) + 1e-10) * 0.95

    # Export to in-memory WAV buffers
    buf = io.BytesIO()
    buf.seek(0)
    # Encode as Base64 for the browser
    b64_n = base64.b64encode(buf.read()).decode("utf-8")
    
    return f"data:audio/wav;base64,{b64_n}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/separate", methods=["POST"])
def separate():
    if "mix1" not in request.files or "mix2" not in request.files:
        return jsonify({"error": "Both audio files are required."}), 400

    file1 = request.files["mix1"]
    file2 = request.files["mix2"]

    if file1.filename == "" or file2.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        audio1_uri, audio2_uri = fast_ica_separate(file1.stream, file2.stream)
        return jsonify({"speaker1": audio1_uri, "speaker2": audio2_uri})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
