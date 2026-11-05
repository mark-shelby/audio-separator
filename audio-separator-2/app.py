import os
import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    # Get the expected number of speakers (n)
    try:
        n_speakers = int(request.form.get('n_speakers', len(files) - 1))
    except ValueError:
        return jsonify({'error': 'Invalid number of speakers.'}), 400
    
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least 2 microphone recordings.'}), 400

    signals = []
    sample_rate = None

    # Read and store all uploaded mic files
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        sr, data = wavfile.read(filepath)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            return jsonify({'error': f'Sample rate mismatch in {filename}. All mics must have the same sample rate.'}), 400
        
        # Convert to mono if the mic recorded in stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        signals.append(data)

    # Trim arrays to the length of the shortest recording to prevent shape errors
    min_len = min(len(sig) for sig in signals)
    signals = [sig[:min_len] for sig in signals]

    # Stack into an Observation Matrix (X)
    # Shape becomes: (n_samples, n_mics)
    X = np.vstack(signals).T

    # Initialize FastICA. It reduces the n+1 mics to n_speakers automatically.
    ica = FastICA(n_components=n_speakers, random_state=42)
    
    # Reconstruct independent signals
    S_ = ica.fit_transform(X)
    
    output_files = []
    # Save the separated components
    for i in range(n_speakers):
        out_name = f'speaker_{i+1}.wav'
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        
        # Normalize audio to standard 16-bit PCM scale to prevent clipping/static
        source_audio = S_[:, i]
        source_audio = np.int16((source_audio / np.max(np.abs(source_audio))) * 32767)
        
        wavfile.write(out_path, sample_rate, source_audio)
        output_files.append(out_name)
        
    return jsonify({'message': 'Success', 'files': output_files})

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)