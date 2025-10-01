import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import whisper
import datetime
import torch
import numpy as np
import wave
import contextlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Speaker Clustering Tool")
root.geometry("500x400")

# Global variables
file_path = ""
num_speakers = tk.IntVar(value=2)
language = tk.StringVar(value="English")
model_size = tk.StringVar(value="medium")

# Function to upload file
def upload_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")])
    if file_path:
        file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

# Function to process the audio
def process_audio():
    global file_path, num_speakers, language, model_size

    if not file_path:
        messagebox.showerror("Error", "Please upload an audio file first!")
        return

    num_spk = num_speakers.get()
    lang = language.get()
    model_sz = model_size.get()

    # Load Whisper model
    model_name = model_sz if lang != 'English' or model_sz == 'large' else f"{model_sz}.en"
    model = whisper.load_model(model_name)

    # Convert to WAV if needed
    if not file_path.endswith('.wav'):
        new_path = "audio.wav"
        subprocess.call(['ffmpeg', '-i', file_path, new_path, '-y'])
        file_path = new_path

    # Transcribe audio
    result = model.transcribe(file_path)
    segments = result["segments"]

    # Get audio duration
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Load Pyannote Speaker Embedding model
    audio = Audio()
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def segment_embedding(segment):
        start, end = segment["start"], min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, _ = audio.crop(file_path, clip)
        return embedding_model(waveform[None])

    # Generate embeddings
    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)

    # Perform clustering
    clustering = AgglomerativeClustering(num_spk).fit(embeddings)
    labels = clustering.labels_

    for i, segment in enumerate(segments):
        segment["speaker"] = f'SPEAKER {labels[i] + 1}'

    def format_time(secs):
        return str(datetime.timedelta(seconds=round(secs)))

    # Write transcript
    with open("transcript.txt", "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write(f"\n{segment['speaker']} {format_time(segment['start'])}\n")
            f.write(segment["text"][1:] + ' ')

    # Display transcript
    messagebox.showinfo("Success", "Speaker Clustering Completed! Check transcript.txt")
    print(open('transcript.txt', 'r').read())

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    unique_speakers = len(set(labels))

    if unique_speakers > 1:
        sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='Set1', s=100, edgecolor='k')
        plt.legend(title="Speakers")
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue', s=100, edgecolor='k', label="Speaker 1")
        plt.legend()

    for i, txt in enumerate(labels):
        plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=10, alpha=0.75)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Speaker Clustering Visualization")
    plt.grid(True)
    plt.show()

# GUI Layout
file_label = tk.Label(root, text="No file selected", wraplength=400)
file_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Audio File", command=upload_file)
upload_button.pack(pady=5)

tk.Label(root, text="Number of Speakers:").pack()
num_speakers_entry = tk.Entry(root, textvariable=num_speakers)
num_speakers_entry.pack(pady=5)

tk.Label(root, text="Select Language:").pack()
lang_dropdown = ttk.Combobox(root, textvariable=language, values=["any", "English"])
lang_dropdown.pack(pady=5)

tk.Label(root, text="Select Model Size:").pack()
model_dropdown = ttk.Combobox(root, textvariable=model_size, values=["tiny", "base", "small", "medium", "large"])
model_dropdown.pack(pady=5)

process_button = tk.Button(root, text="Perform Speaker Clustering", command=process_audio)
process_button.pack(pady=10)

# Start the GUI
root.mainloop()
