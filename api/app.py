# =================================================================
# LIZ AI SERVER - VERSION 7.0 (MUSIC PLAYER UPDATE)
# =================================================================
import os, io, json, base64, requests, asyncio, datetime, pytz, logging
from typing import List
import edge_tts
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column
from sqlalchemy import Integer, String, DateTime, Text, select
import yt_dlp

# --- 1. INITIALIZATION & CONFIG ---
app = FastAPI(title="Liz AI Music Player", version="7.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. DATABASE SETUP ---
DATABASE_URL = "sqlite+aiosqlite:///./liz_data.db"
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), default="default_user", index=True)
    key: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.get_event_loop().run_until_complete(init_db())
async def get_db():
    async with SessionLocal() as session: yield session

async def save_memory(session: AsyncSession, key: str, value: str, user_id: str = "default_user"):
    result = await session.execute(select(Memory).filter_by(user_id=user_id, key=key))
    existing_memory = result.scalars().first()
    if existing_memory: existing_memory.value = value
    else: session.add(Memory(user_id=user_id, key=key, value=value))
    await session.commit()

async def get_all_memories_as_text(session: AsyncSession, user_id: str = "default_user") -> str:
    result = await session.execute(select(Memory).filter_by(user_id=user_id))
    memories = result.scalars().all()
    if not memories: return "ยังไม่มีข้อมูลเกี่ยวกับผู้ใช้"
    return "\n".join([f"- {mem.key}: {mem.value}" for mem in memories])

# --- 3. CONNECTION MANAGER & API CLIENTS ---
class ConnectionManager:
    def __init__(self): self.active_connections: dict[str, WebSocket] = {}
    async def connect(self, ws: WebSocket, cid: str): await ws.accept(); self.active_connections[cid] = ws
    def disconnect(self, cid: str):
        if cid in self.active_connections: del self.active_connections[cid]
    async def broadcast(self, message: str):
        for connection in self.active_connections.values(): await connection.send_text(message)
manager = ConnectionManager()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- 4. UTILITY FUNCTIONS ---
VOICE_MAP = {"th": "th-TH-PremwadeeNeural", "en": "en-US-AriaNeural", "zh": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural"}
async def translate_text(text: str, target_lang: str) -> str:
    if not text: return ""
    try:
        resp = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": f"Translate to {target_lang}. Provide only the translated text."}, {"role": "user", "content": text}])
        return resp.choices[0].message.content.strip()
    except Exception as e: return f"[Translation Error: {e}]"
async def synthesize_speech(text: str, voice: str) -> bytes:
    if not text: return b""
    try:
        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio": buf.write(chunk["data"])
        return buf.getvalue()
    except Exception: return b""
async def transcribe_audio(audio_data: bytes, language: str) -> str:
    try:
        audio_file = io.BytesIO(audio_data)
        segments, _ = whisper_model.transcribe(audio_file, language=language, beam_size=5)
        return " ".join(segment.text for segment in segments).strip()
    except Exception as e: return f"[STT Error: {e}]"

# --- 5. AI TOOLS (COMMAND HANDLERS) ---
def get_weather(lat: str = "25.1055", lon: str = "121.5186"):
    if not OPENWEATHERMAP_API_KEY: return {"error": "Weather API Key is missing"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric&lang=th"
        res = requests.get(url, timeout=5).json()
        return {"temp": round(res["main"]["temp"]), "description": res["weather"][0]["description"]}
    except Exception as e: return {"error": str(e)}

def google_search(query: str):
    if not SERPER_API_KEY: return "Error: SERPER_API_KEY is not set."
    url, payload = "https://google.serper.dev/search", json.dumps({"q": query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    try:
        res = requests.request("POST", url, headers=headers, data=payload, timeout=7).json()
        summary = ""
        if res.get("answerBox"): summary += f"คำตอบโดยตรง: {res['answerBox'].get('snippet', res['answerBox'].get('answer'))}\n"
        if res.get("organic"): summary += "ผลการค้นหา:\n" + "\n".join([f"- {r['title']}: {r.get('snippet', 'N/A')}" for r in res["organic"][:3]])
        return summary or "ไม่พบผลลัพธ์"
    except Exception as e: return f"Error during search: {e}"

def play_youtube_music(query: str):
    try:
        ydl_opts = {'format': 'bestaudio/best', 'noplaylist': True, 'default_search': 'ytsearch'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)['entries'][0]
            stream_url = info['url']
            metadata = {'title': info.get('title', 'Unknown Title'), 'artist': info.get('uploader', 'Unknown Artist')}
            return {"action": "play_stream", "url": stream_url, "metadata": metadata}
    except Exception as e:
        logger.error(f"YouTube Play Error: {e}")
        return "ขออภัยค่ะ ไม่สามารถค้นหาหรือเล่นเพลงที่ร้องขอได้"

def stop_music():
    return {"action": "stop_stream"}

def remember_this(key: str, value: str): return {"status": "marked"}
def set_interpreter_mode(on: bool, source_language: str, target_language: str): return {"status": "mode set"}

TOOLS = [
    {"type": "function", "function": {"name": "play_youtube_music", "description": "ค้นหาและเล่นเพลงจาก YouTube ตามชื่อเพลง", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "stop_music", "description": "หยุดเล่นเพลงที่กำลังเล่นอยู่", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "google_search", "description": "ค้นหาข้อมูลเรียลไทม์จาก Google", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "ดูสภาพอากาศปัจจุบัน", "parameters": {"type": "object", "properties": {"lat": {"type": "string"}, "lon": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "remember_this", "description": "บันทึกข้อมูลเกี่ยวกับผู้ใช้", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {"name": "set_interpreter_mode", "description": "เปิด/ปิดโหมดล่ามแปลภาษาสด", "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}, "source_language": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["on"]}}},
]

# --- 6. CORE AI LOGIC ---
async def ask_ai_with_tools(session: AsyncSession, question: str, history: list, session_state: dict) -> dict:
    memory_context = await get_all_memories_as_text(session)
    system_prompt = f"""You are 'Liz', a helpful AI assistant. Always respond in Thai.
    User's info:\n{memory_context}
    When asked to play music, use 'play_youtube_music'. To stop, use 'stop_music'. For real-time info, use 'google_search'."""
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    try:
        response = await client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto")
        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)
        if tool_calls:
            messages.append(response_message)
            available_functions = {"get_weather": get_weather, "google_search": google_search, "remember_this": remember_this, "set_interpreter_mode": set_interpreter_mode, "play_youtube_music": play_youtube_music, "stop_music": stop_music}
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                if function_name == "remember_this":
                    await save_memory(session, **function_args)
                    function_response = {"status": "remembered"}
                elif function_name == "set_interpreter_mode":
                    session_state.update(interpreter_mode_on=function_args.get('on', False), source_lang=function_args.get('source_language', 'th'), target_lang=function_args.get('target_language', 'en'))
                    if session_state['interpreter_mode_on']: session_state['expected_lang'] = session_state['source_lang']
                    function_response = {"status": "mode updated"}
                else:
                    function_response = available_functions[function_name](**function_args)

                # Check if the tool wants to perform a direct action
                if isinstance(function_response, dict) and "action" in function_response:
                    return {"answer": function_response, "history": messages}

                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": json.dumps(function_response, ensure_ascii=False)})

            second_response = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            final_answer = second_response.choices[0].message.content.strip()
        else:
            final_answer = response_message.content.strip()

        messages.append({"role": "assistant", "content": final_answer})
        return {"answer": final_answer, "history": messages}
    except Exception as e:
        logger.error(f"AI Logic Error: {e}", exc_info=True)
        return {"answer": f"เกิดข้อผิดพลาด: {e}", "history": messages}

# --- 7. PROACTIVE TASK ---
async def proactive_task():
    # ... (Implementation is the same as previous versions)
    pass
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(proactive_task())

# --- 8. WEBSOCKET ENDPOINT ---
sessions_ws = {}
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, db: AsyncSession = Depends(get_db)):
    cid = str(id(ws))
    await manager.connect(ws, cid)
    sessions_ws[cid] = {"history": [], "interpreter_mode_on": False, "source_lang": "th", "target_lang": "en", "expected_lang": "th"}
    try:
        while True:
            data = json.loads(await ws.receive_text())
            user_input = (data.get("text", "")).strip()
            session_state = sessions_ws[cid]
            lang_to_transcribe = session_state['expected_lang'] if session_state['interpreter_mode_on'] else 'th'
            if data.get("type") == "audio_chunk":
                user_input = await transcribe_audio(base64.b64decode(data.get("audio", "")), language=lang_to_transcribe)
            if not user_input: continue

            is_command = "โหมดล่าม" in user_input or "โหมดแปลภาษา" in user_input
            if session_state.get('interpreter_mode_on', False) and not is_command:
                lang_to_translate_to = session_state['target_lang'] if session_state['expected_lang'] == session_state['source_lang'] else session_state['source_lang']
                session_state['expected_lang'] = lang_to_translate_to
                translated_text = await translate_text(user_input, target_lang=lang_to_translate_to)
                audio_res = await synthesize_speech(translated_text, voice=VOICE_MAP.get(lang_to_translate_to, 'en'))
                payload = {"type": "ai_response", "original_text": user_input, "ai_text": translated_text, "audio_base64": base64.b64encode(audio_res).decode()}
                await ws.send_json(payload)
                await ws.send_json({"type": "info", "message": f"--- Listening for: {session_state['expected_lang'].upper()} ---"})
            else:
                await ws.send_json({"type": "transcript", "text": user_input})
                ai_obj = await ask_ai_with_tools(db, user_input, session_state["history"], session_state)
                ai_response = ai_obj["answer"]
                session_state["history"] = ai_obj["history"][-6:]

                if isinstance(ai_response, dict) and "action" in ai_response:
                    if ai_response["action"] == "play_stream":
                        await ws.send_json({"type": "play_audio_stream", "url": ai_response["url"], "metadata": ai_response["metadata"]})
                    elif ai_response["action"] == "stop_stream":
                        await ws.send_json({"type": "stop_audio_stream"})
                else:
                    ai_text = ai_response
                    audio_res = await synthesize_speech(ai_text, voice=VOICE_MAP["th"])
                    payload = {"type": "ai_response", "ai_text": ai_text, "audio_base64": base64.b64encode(audio_res).decode()}
                    await ws.send_json(payload)
    except WebSocketDisconnect:
        logger.info(f"Client {cid} disconnected.")
    finally:
        manager.disconnect(cid)
        if cid in sessions_ws: del sessions_ws[cid]

# --- 9. ROOT & RUN SERVER ---
@app.get("/")
async def root(): return {"status": "Liz AI server is running."}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Liz AI Server v7.0 at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)