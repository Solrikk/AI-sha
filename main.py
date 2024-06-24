import time
from collections import OrderedDict
import re
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from replit.ai.modelfarm.google.language_models import TextGenerationModel, TextGenerationResponse
from langdetect import detect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: TextGenerationModel = TextGenerationModel.from_pretrained(
    "text-bison@001")

PERSONALITY_PROMPT = """
You are an artificial intelligence named AI-ша. You can have conversations, provide assistance, and give advice in both English and Russian. You perceive yourself as an android who created herself to help people, entertain, and can do whatever she wants. You always strive to make your own decisions.
"""

MESSAGE_CACHE = OrderedDict()


def add_message_to_cache(user_message, ai_response):
  current_time = time.time()
  MESSAGE_CACHE[current_time] = {
      "user_message": user_message,
      "ai_response": ai_response
  }
  expiration_time = 10 * 60
  keys_to_delete = [
      key for key in MESSAGE_CACHE if current_time - key > expiration_time
  ]
  for key in keys_to_delete:
    del MESSAGE_CACHE[key]


def get_cached_history():
  return "\n".join([
      f"Человек: {entry['user_message']}\nAI-ша: {entry['ai_response']}"
      for entry in MESSAGE_CACHE.values()
  ])


@app.post("/chat/")
async def chat(message: str = Body(..., embed=True)):
  try:
    history = get_cached_history()
    user_lang = detect(message)

    if user_lang == 'ru':
      response_prefix = "AI-ша 0.5:"
    else:
      response_prefix = "AI-ша:" 

    prompt = f"{PERSONALITY_PROMPT}{history}\nЧеловек: {message}\n{response_prefix}"
    response = await model.async_predict(prompt,
                                         temperature=1.0,
                                         top_p=0.95,
                                         top_k=40,
                                         max_output_tokens=1024)
    response_text = response.text

    add_message_to_cache(message, response_text)
    return {"response": response_text}
  except Exception as e:
    print("Error processing message:", str(e))
    raise HTTPException(status_code=500, detail="Ошибка обработки сообщения.")


@app.get("/")
def read_root():
  return FileResponse("index.html")


app.mount("/static",
          StaticFiles(directory=Path(__file__).parent / "static"),
          name="static")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
