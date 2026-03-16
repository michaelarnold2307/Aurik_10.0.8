# REST-API für Deep-Learning Artefakt-Detection in Aurik
# FastAPI, Batch & Echtzeit

from fastapi import FastAPI, File, UploadFile
import soundfile as sf

from plugins.artifact_detection_plugin import ArtifactDetectionPlugin

app = FastAPI()
plugin = ArtifactDetectionPlugin("models/artifact_detector.pt")


@app.post("/detect-artifacts")
def detect_artifacts(file: UploadFile = File(...)):
    data, sr = sf.read(file.file)
    result = plugin.detect_artifacts(data, sr)
    return result


@app.post("/feedback")
def feedback(feedback: dict):
    plugin.feedback(feedback)
    return {"status": "received"}


# Beispiel: curl -X POST -F 'file=@audio.wav' http://localhost:8000/detect-artifacts
