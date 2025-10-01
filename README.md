# 🎙️ Speaker-Clustering Tool

An interactive **Speaker Clustering and Transcription Tool** that uses **OpenAI Whisper** for transcription and **Pyannote Audio** for speaker embeddings. This project allows you to:

* Upload an audio file (`.wav`, `.mp3`, `.m4a`)
* Automatically **transcribe speech** into text
* Identify and **cluster multiple speakers** in the conversation
* Save the transcript with speaker labels
* Visualize the **speaker embeddings** using PCA

Built with **Tkinter GUI**, so no command-line hassle 🚀

---

## ✨ Features

✅ Upload audio files easily via GUI
✅ Choose Whisper **model size** (`tiny`, `base`, `small`, `medium`, `large`)
✅ Select language preference (`English` or `any`)
✅ Specify number of speakers for clustering
✅ Generates a **speaker-labeled transcript** (`transcript.txt`)
✅ PCA-based **visualization of speaker clusters**

---

## 📦 Dependencies

Make sure you have the following installed before running:

```
pip install openai-whisper pyannote.audio torch numpy matplotlib seaborn scikit-learn tk
```

You also need **FFmpeg** installed for audio conversion:

* [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.

---

## 🚀 Usage

1. Clone this repository:

   ```
   git clone https://github.com/NikhTheTech/Speaker-Clustering.git
   cd Speaker-Clustering
   ```

2. Run the tool:

   ```
   python GUI_SC.py
   ```

3. In the GUI:

   * Upload your audio file
   * Select number of speakers
   * Choose language and Whisper model size
   * Click **Perform Speaker Clustering**

4. Output:

   * `transcript.txt` → contains transcribed text with speaker labels
   * A PCA visualization plot → shows clustered speaker embeddings

---

## 📂 Example Output

**Transcript (`transcript.txt`):**

```
SPEAKER 1 00:00:05
Hello everyone, welcome to the session.

SPEAKER 2 00:00:12
Thank you! Glad to be here.
```

**Visualization:**
The tool plots clustered embeddings in 2D space:

* Each point = audio segment
* Colors = speaker identity

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **Whisper** (speech recognition)
* **Pyannote Audio** (speaker embeddings)
* **Agglomerative Clustering** (speaker grouping)
* **Tkinter** (GUI)
* **Matplotlib + Seaborn** (visualization)

---

## ⚡ Future Improvements

* Automatic **speaker count estimation**
* Export transcript in **SRT/JSON format**
* Option to process **longer audio files** with segmentation
* Integration with **real-time audio streams**
