import os
import whisper
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

MODEL_SIZE = "small"
LANGUAGE = "ja"

TERMINOLOGY_DICT = {
    "バブル": "バルブ",
    "グローブバルブ": "グローブ弁",
    "チェックバルブ": "逆止弁",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model(MODEL_SIZE)

def correct(text):
    for k, v in TERMINOLOGY_DICT.items():
        text = text.replace(k, v)
    return text

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    result = model.transcribe(
        tmp_path,
        language=LANGUAGE,
        fp16=False
    )

    os.remove(tmp_path)

    text = "\n".join(correct(s["text"]) for s in result["segments"])
    return {"text": text}
