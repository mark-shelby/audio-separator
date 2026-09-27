import base64
import io
from flask import Flask, jsonify, render_template, request
import numpy as np
import soundfile as sf

app = Flask(__name__)


def fast_ica_separate(file1_stream, file2_stream):
    # Read audio from memory streams
    x1, sr1 = sf.read(file1_stream)
    x2, sr2 = sf.read(file2_stream)

    # Convert stereo to mono if necessary
    if x1.ndim > 1:
        x1 = np.mean(x1, axis=1)
    if x2.ndim > 1:
        x2 = np.mean(x2, axis=1)

    # Truncate to shortest length
    min_len = min(len(x1), len(x2))
    x1 = x1[:min_len]
    x2 = x2[:min_len]

    # Mix matrix
    X = np.vstack((x1, x2)).T

    # Mean centering
    X = X - np.mean(X, axis=0)

    # Whitening
    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    Z = (eigenvectors @ D_inv_sqrt @ eigenvectors.T @ X.T).T

    # FastICA decomposition
    W = np.zeros((2, 2))
    for i in range(2):
        w = np.random.rand(2)
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
    s1 = S[:, 0] / (np.max(np.abs(S[:, 0])) + 1e-10) * 0.95
    s2 = S[:, 1] / (np.max(np.abs(S[:, 1])) + 1e-10) * 0.95

    # Export to in-memory WAV buffers
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    sf.write(buf1, s1, sr1, format="WAV")
    sf.write(buf2, s2, sr1, format="WAV")
    buf1.seek(0)
    buf2.seek(0)

    # Encode as Base64 for the browser
    b64_1 = base64.b64encode(buf1.read()).decode("utf-8")
    b64_2 = base64.b64encode(buf2.read()).decode("utf-8")

    return f"data:audio/wav;base64,{b64_1}", f"data:audio/wav;base64,{b64_2}"


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