# =================================================================
# LIZ AI SERVER - VERSION 7.2 (PRODUCTION-READY)
# =================================================================
import os, io, json, base64, asyncio, logging, time
from pathlib import Path
from typing import List

# --- Transformers & Hugging Face ---
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download, HfHubHTTPError

# --- TTS & Music ---
import edge_tts
import yt_dlp

# --- FastAPI ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# --- Database ---
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, select

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LizAI")

# =================================================================
# 1. FASTAPI & CORS
# =================================================================
app = FastAPI(title="Liz AI Music Player", version="7.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# 2. DATABASE SETUP
# =================================================================
DATABASE_URL = "sqlite+aiosqlite:////tmp/lizai_prod.db"
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(engine,
                            expire_on_commit=False,
                            class_=AsyncSession)
Base = declarative_base()


class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100),
                                         default="default_user",
                                         index=True)
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


asyncio.get_event_loop().run_until_complete(init_db())


async def get_db():
    async with SessionLocal() as session:
        yield session


async def save_memory(session: AsyncSession,
                      key: str,
                      value: str,
                      user_id: str = "default_user"):
    result = await session.execute(
        select(Memory).filter_by(user_id=user_id, key=key))
    existing_memory = result.scalars().first()
    if existing_memory:
        existing_memory.value = value
    else:
        session.add(Memory(user_id=user_id, key=key, value=value))
    await session.commit()


async def get_all_memories_as_text(session: AsyncSession,
                                   user_id: str = "default_user") -> str:
    result = await session.execute(select(Memory).filter_by(user_id=user_id))
    memories = result.scalars().all()
    if not memories: return "ยังไม่มีข้อมูลเกี่ยวกับผู้ใช้"
    return "\n".join([f"- {mem.key}: {mem.value}" for mem in memories])


# =================================================================
# 3. CONNECTION MANAGER
# =================================================================
class ConnectionManager:

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, ws: WebSocket, cid: str):
        await ws.accept()
        self.active_connections[cid] = ws

    def disconnect(self, cid: str):
        if cid in self.active_connections:
            del self.active_connections[cid]

    async def broadcast(self, message: str):
        for ws in self.active_connections.values():
            await ws.send_text(message)


manager = ConnectionManager()

# =================================================================
# 4. CONFIG & HF TOKEN
# =================================================================
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing!")

VOICE_MAP = {
    "th": "th-TH-PremwadeeNeural",
    "en": "en-US-AriaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural"
}

# =================================================================
# 5. LAZY LOAD LLM & WHISPER + RETRY
# =================================================================
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
llm_pipeline = None
whisper_model = None


def retry_download(repo_id, retries=3, wait=5):
    for i in range(retries):
        try:
            path = snapshot_download(repo_id, use_auth_token=HF_TOKEN)
            return path
        except HfHubHTTPError as e:
            logger.warning(
                f"Hugging Face download failed ({i+1}/{retries}): {e}")
            time.sleep(wait)
    raise RuntimeError(
        f"Cannot download model {repo_id} after {retries} retries")


def load_llm_pipeline():
    global llm_pipeline
    if llm_pipeline is None:
        logger.info(f"🚀 Loading LLM pipeline: {LLM_MODEL_NAME}")
        retry_download(LLM_MODEL_NAME)
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME,
                                                     use_auth_token=HF_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME,
                                                  use_auth_token=HF_TOKEN)
        llm_pipeline = pipeline("text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                device=-1)
        logger.info("✅ LLM Pipeline loaded successfully")
    return llm_pipeline


def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        logger.info("🚀 Loading Faster Whisper model...")
        retry_download("openai/whisper-base")  # หรือชื่อโมเดลจริง
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base",
                                     device="cpu",
                                     compute_type="int8",
                                     download_root="/tmp",
                                     local_files_only=False)
        logger.info("✅ Faster Whisper loaded successfully")
    return whisper_model


async def transcribe_audio(audio_data: bytes, language: str = "th") -> str:
    model = load_whisper_model()
    audio_file = io.BytesIO(audio_data)
    segments, _ = model.transcribe(audio_file, language=language, beam_size=5)
    return " ".join(segment.text for segment in segments).strip()


async def translate_text(text: str, target_lang: str) -> str:
    if not text: return ""
    try:
        llm = load_llm_pipeline()
        prompt = f"Translate the following text to {target_lang}. Only provide translated text: {text}"
        response = await asyncio.to_thread(llm,
                                           prompt,
                                           max_new_tokens=100,
                                           do_sample=False)
        return response[0]["generated_text"].replace(prompt, "").strip()
    except Exception as e:
        logger.error(f"Translation Error: {e}")
        return f"[Translation Error: {e}]"


async def synthesize_speech(text: str, voice: str) -> bytes:
    if not text: return b""
    try:
        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()
    except Exception:
        return b""


# =================================================================
# 6. MUSIC & COMMANDS
# =================================================================
def play_youtube_music(query: str):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'default_search': 'ytsearch'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)['entries'][0]
            return {
                "action": "play_stream",
                "url": info['url'],
                "metadata": {
                    "title": info.get('title'),
                    "artist": info.get('uploader')
                }
            }
    except Exception as e:
        logger.error(f"YouTube Play Error: {e}")
        return "ไม่สามารถเล่นเพลงได้"


def stop_music():
    return {"action": "stop_stream"}


# =================================================================
# 7. AI CORE LOGIC
# =================================================================
async def ask_ai_with_tools(session: AsyncSession, question: str,
                            history: list, session_state: dict) -> dict:
    global llm_pipeline
    if llm_pipeline is None:
        llm_pipeline = await asyncio.to_thread(load_llm_pipeline)
    q_lower = question.lower()
    if "เล่นเพลง" in q_lower or "เปิดเพลง" in q_lower:
        return {"answer": play_youtube_music(question), "history": history}
    if "หยุดเพลง" in q_lower:
        return {"answer": stop_music(), "history": history}

    memory_context = await get_all_memories_as_text(session)
    system_prompt = f"You are 'Liz', a helpful AI assistant. Always respond in Thai.\nUser's info:\n{memory_context}"

    full_prompt = f"### System Instruction:\n{system_prompt}\n\n"
    for msg in history:
        role = "Assistant" if msg["role"] == "assistant" else "User"
        full_prompt += f"### {role}: {msg['content']}\n"
    full_prompt += f"### User: {question}\n### Assistant: "

    try:
        response = await asyncio.to_thread(llm_pipeline,
                                           full_prompt,
                                           max_new_tokens=256,
                                           do_sample=True,
                                           temperature=0.7,
                                           return_full_text=False)
        final_answer = response[0]['generated_text'].strip()
        messages = history + [{
            "role": "user",
            "content": question
        }, {
            "role": "assistant",
            "content": final_answer
        }]
        return {"answer": final_answer, "history": messages[-6:]}
    except Exception as e:
        logger.error(f"AI Logic Error: {e}", exc_info=True)
        return {"answer": f"เกิดข้อผิดพลาด: {e}", "history": history}


# =================================================================
# 8. WEBSOCKET
# =================================================================
sessions_ws = {}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket,
                             db: AsyncSession = Depends(get_db)):
    cid = str(id(ws))
    await manager.connect(ws, cid)
    sessions_ws[cid] = {
        "history": [],
        "interpreter_mode_on": False,
        "source_lang": "th",
        "target_lang": "en",
        "expected_lang": "th"
    }
    try:
        while True:
            data = json.loads(await ws.receive_text())
            user_input = (data.get("text", "")).strip()
            session_state = sessions_ws[cid]
            if data.get("type") == "audio_chunk":
                audio_bytes = base64.b64decode(data.get("audio", ""))
                user_input = await transcribe_audio(
                    audio_bytes, language=session_state['expected_lang'])
            if not user_input: continue

            is_command = "โหมดล่าม" in user_input or "โหมดแปลภาษา" in user_input
            if session_state.get("interpreter_mode_on",
                                 False) and not is_command:
                lang_to_translate_to = session_state[
                    'target_lang'] if session_state[
                        'expected_lang'] == session_state[
                            'source_lang'] else session_state['source_lang']
                session_state['expected_lang'] = lang_to_translate_to
                translated_text = await translate_text(
                    user_input, target_lang=lang_to_translate_to)
                audio_res = await synthesize_speech(translated_text,
                                                    voice=VOICE_MAP.get(
                                                        lang_to_translate_to,
                                                        "en"))
                await ws.send_json({
                    "type":
                    "ai_response",
                    "original_text":
                    user_input,
                    "ai_text":
                    translated_text,
                    "audio_base64":
                    base64.b64encode(audio_res).decode()
                })
                await ws.send_json({
                    "type":
                    "info",
                    "message":
                    f"--- Listening for: {session_state['expected_lang'].upper()} ---"
                })
            else:
                await ws.send_json({"type": "transcript", "text": user_input})
                ai_obj = await ask_ai_with_tools(db, user_input,
                                                 session_state["history"],
                                                 session_state)
                ai_response = ai_obj["answer"]
                session_state["history"] = ai_obj["history"][-6:]
                if isinstance(ai_response, dict) and "action" in ai_response:
                    if ai_response["action"] == "play_stream":
                        await ws.send_json({
                            "type": "play_audio_stream",
                            "url": ai_response["url"],
                            "metadata": ai_response["metadata"]
                        })
                    elif ai_response["action"] == "stop_stream":
                        await ws.send_json({"type": "stop_audio_stream"})
                else:
                    ai_text = ai_response
                    audio_res = await synthesize_speech(ai_text,
                                                        voice=VOICE_MAP["th"])
                    await ws.send_json({
                        "type":
                        "ai_response",
                        "ai_text":
                        ai_text,
                        "audio_base64":
                        base64.b64encode(audio_res).decode()
                    })
    except WebSocketDisconnect:
        logger.info(f"Client {cid} disconnected.")
    finally:
        manager.disconnect(cid)
        if cid in sessions_ws: del sessions_ws[cid]


# =================================================================
# 9. STATIC FILES & ROOT
# =================================================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "public"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_root():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return {"status": "Liz AI running. (No index.html)"}
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/status")
async def get_status():
    return {"status": "Liz AI server is running."}


# =================================================================
# 10. STARTUP EVENT
# =================================================================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Liz AI Server v7.2 Production starting...")


# =================================================================
# 11. RUN (FOR DEV; Production uses Gunicorn)
# =================================================================
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Running Liz AI Server locally at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
