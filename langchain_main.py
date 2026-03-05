"""
Voice Assistant - Cleanly Separated Architecture
================================================
FIX: Improved follow-up detection — now catches imperative prompts like
     "Please tell me your roll number." (no ? required).
"""

import openwakeword
from openwakeword.model import Model
import pyaudio
import numpy as np
import time
import asyncio
import websockets
import json
import base64
import wave
import io
from collections import deque
from threading import Thread, Event
import pygame
import webrtcvad
from app.tts import text_to_speech, switch_tts_language
from app.config.config import CURRENT_LANGUAGE, LANGUAGE_NAMES
import uuid



# ============================================================
# CONFIGURATION
# ============================================================


# Wake Word
MODEL_PATH           = "app/onnx/up_tal.onnx"
DETECTION_THRESHOLD  = 0.0015
ENABLE_SPEEX         = False
VAD_THRESHOLD        = None
SMOOTHING_WINDOW     = 5
COOLDOWN_TIME        = 3.0
INSTANT_SCORE_GATE   = 0.001

# Audio
CHUNK_SIZE = 1280
FORMAT     = pyaudio.paInt16
CHANNELS   = 1
RATE       = 16000

# Sound file paths
RESPONSE_AUDIO_PATH = "app/sounds/ding.mp3"
HITL_CONFIRM_SOUND  = "app/sounds/response_user.wav"
HITL_REVIEW_SOUND   = "app/sounds/notification.mp3"
INSTRUCTION_SOUND   = "app/sounds/say_uptal.wav"
WAIT_MOMENT_SOUND   = "app/sounds/wait_moment.wav"
THINKING_SOUND      = "app/sounds/thinking.mp3"

# WebRTC VAD
VAD_MODE           = 3
VAD_FRAME_DURATION = 30
VAD_FRAME_SIZE     = int(RATE * VAD_FRAME_DURATION / 1000)

# Recording
RECORDING_DURATION = 10
MIN_RECORDING_TIME = 1.5
SILENCE_DURATION   = 1.5
VAD_SILENCE_CHUNKS = int(SILENCE_DURATION * 1000 / VAD_FRAME_DURATION)

# Validation
MIN_SPEECH_FRAMES_REQUIRED = 3
MIN_AUDIO_AMPLITUDE        = 50
MIN_SPEECH_PERCENTAGE      = 0.08

# WebSocket
WEBSOCKET_URI = "ws://localhost:8765"

# HITL
HITL_ENABLED = True


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

import re

# Patterns that mean the assistant is waiting for user input
_FOLLOWUP_QUESTION_MARKS = ("?", "؟", "？")   # literal question marks

_FOLLOWUP_IMPERATIVE = re.compile(
    r'\b(tell me|give me|provide|say|speak|enter|type|share|mention|'
    r'repeat|confirm|let me know|what is|what\'s|please|bata|batao|'
    r'dijiye|batayein|please tell|please give|please say)\b',
    re.I
)





# Phrases that clearly end the conversation (no follow-up needed)
_TERMINAL_PHRASES = re.compile(
    r'\b(done|completed|set|deleted|sent|found|here (is|are)|'
    r'your reminder|reminder set|result is|grade is|marks are|'
    r'i (have|\'ve) (set|found|sent|deleted|created))\b',
    re.I
)



def _needs_followup(text: str) -> bool:
    """
    Return True if the assistant's response expects user input.
    Catches both '?' endings AND imperative prompts like
    'Please tell me your roll number.'
    """
    t = text.strip()

    # 1. Explicit question mark (any script)
    if any(t.endswith(q) for q in _FOLLOWUP_QUESTION_MARKS):
        return True

    # 2. Ends with a question mark somewhere inside (embedded question)
    if any(q in t for q in _FOLLOWUP_QUESTION_MARKS):
        return True

    # 3. Imperative / request phrase — but NOT if it also looks terminal
    if _FOLLOWUP_IMPERATIVE.search(t) and not _TERMINAL_PHRASES.search(t):
        return True

    return False


# ============================================================
# LANGUAGE SWITCH HELPER
# ============================================================

def switch_language(lang_code: str):
    app.config.config.CURRENT_LANGUAGE = lang_code
    print(f"[LANG] Switched to: {LANGUAGE_NAMES.get(lang_code, lang_code)}")
    switch_tts_language(lang_code)


# ============================================================
# SHARED VAD HELPERS
# ============================================================

_vad = webrtcvad.Vad(VAD_MODE)


def _is_speech(audio_frame) -> bool:
    try:
        raw = audio_frame.tobytes() if isinstance(audio_frame, np.ndarray) else audio_frame
        return _vad.is_speech(raw, RATE)
    except Exception:
        arr = np.frombuffer(audio_frame, dtype=np.int16) if isinstance(audio_frame, bytes) else audio_frame
        return float(np.abs(arr).mean()) > 300


def _check_vad_buffer(audio_array: np.ndarray, num_frames: int = 2) -> bool:
    speech, total = 0, 0
    for i in range(0, len(audio_array) - VAD_FRAME_SIZE, VAD_FRAME_SIZE):
        frame = audio_array[i:i + VAD_FRAME_SIZE]
        if len(frame) == VAD_FRAME_SIZE:
            speech += _is_speech(frame)
            total  += 1
            if total >= num_frames:
                break
    return (speech > total / 2) if total else False


def _validate_recording(audio_bytes: bytes):
    if not audio_bytes:
        return False, "Empty recording", {}
    arr       = np.frombuffer(audio_bytes, dtype=np.int16)
    amplitude = float(np.abs(arr).mean())
    max_amp   = float(np.abs(arr).max())
    stats     = {"duration": len(arr) / RATE, "mean_amplitude": amplitude, "max_amplitude": max_amp}
    if amplitude < MIN_AUDIO_AMPLITUDE:
        return False, f"Audio too quiet ({amplitude:.1f})", stats
    speech_frames = total_frames = 0
    for i in range(0, len(arr) - VAD_FRAME_SIZE, VAD_FRAME_SIZE):
        frame = arr[i:i + VAD_FRAME_SIZE]
        if len(frame) == VAD_FRAME_SIZE:
            speech_frames += _is_speech(frame)
            total_frames  += 1
    speech_pct = speech_frames / total_frames if total_frames else 0
    stats.update({"speech_frames": speech_frames, "total_frames": total_frames,
                  "speech_percentage": speech_pct * 100})
    if speech_frames < MIN_SPEECH_FRAMES_REQUIRED:
        return False, f"Too few speech frames ({speech_frames}/{MIN_SPEECH_FRAMES_REQUIRED})", stats
    if speech_pct < MIN_SPEECH_PERCENTAGE:
        return False, f"Low speech ratio ({speech_pct*100:.1f}%)", stats
    return True, "Valid speech detected", stats


def _audio_to_base64_wav(audio_bytes: bytes) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ============================================================
# THINKING SOUND PLAYER
# ============================================================

class ThinkingPlayer:
    def __init__(self):
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self):
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="ThinkingPlayer")
        self._thread.start()
        print("🔄 Thinking sound started (looping)…")

    def stop(self):
        if not self._stop_event.is_set():
            self._stop_event.set()
            try:
                pygame.mixer.Channel(1).stop()
            except Exception:
                pass
            if self._thread:
                self._thread.join(timeout=3)
            print("⏹️  Thinking sound stopped")

    def _loop(self):
        try:
            sound   = pygame.mixer.Sound(THINKING_SOUND)
            channel = pygame.mixer.Channel(1)
            while not self._stop_event.is_set():
                channel.play(sound)
                while channel.get_busy() and not self._stop_event.is_set():
                    time.sleep(0.05)
        except Exception as e:
            print(f"[ThinkingPlayer] Error: {e}")


_thinking_player = ThinkingPlayer()


# ============================================================
# HITL REVIEW — voice-based
# ============================================================

def _hitl_review_voice(pending_tool_calls: list, recorder) -> tuple[str, str | None]:
    summary_parts = []
    for tc in pending_tool_calls:
        try:
            args = json.loads(tc.get("arguments", "{}") or "{}")
        except Exception:
            args = {}
        args_spoken = ", ".join(f"{k} {v}" for k, v in args.items())
        summary_parts.append(f"{tc['name'].replace('_', ' ')}: {args_spoken}")

    HITL_PROMPTS = {
        "en": "I want to run: {summary}. Say yes to confirm or no to cancel.",
        "hi": "मैं यह करना चाहता हूं: {summary}. हाँ कहें या रद्द करने के लिए नहीं।",
    }

    print("\n🔊 Speaking HITL prompt…")
    prompt_text = HITL_PROMPTS.get(
        CURRENT_LANGUAGE, HITL_PROMPTS["en"]
    ).format(summary=". ".join(summary_parts))

    audio = text_to_speech(prompt_text)
    if audio:
        sound = pygame.mixer.Sound(io.BytesIO(audio))
        sound.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)

    try:
        beep = pygame.mixer.Sound(HITL_REVIEW_SOUND)
        beep.play()
        while pygame.mixer.get_busy():
            time.sleep(0.05)
    except Exception as e:
        print(f"[HITL] Beep error: {e}")

    print("🎙️  Listening for your decision (yes/no)...")
    voice_bytes = recorder.record()

    if not voice_bytes:
        print("   No voice detected — defaulting to execute")
        return "execute", None

    import tempfile, os
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(voice_bytes)
    buf.seek(0)

    tmp = tempfile.mktemp(suffix=".wav")
    with open(tmp, "wb") as f:
        f.write(buf.read())

    try:
        with open(tmp, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3", file=af,
                response_format="text",
                language=CURRENT_LANGUAGE
            )
        decision_text = result.lower().strip()
        print(f"   🎤 Voice decision: '{decision_text}'")
    except Exception as e:
        print(f"   Error transcribing decision: {e}")
        decision_text = "yes"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    CONFIRM_WORDS = {
        "en": ["yes", "execute", "confirm", "ok", "okay", "sure", "go"],
        "hi": ["हाँ", "हां", "ठीक", "करो", "yes", "यश", "यस"],
    }
    CANCEL_WORDS = {
        "en": ["no", "cancel", "discard", "stop", "abort"],
        "hi": ["नहीं", "रुको", "रद्द", "no", "स्टॉप", "स्टाप"],
    }

    confirms = CONFIRM_WORDS.get(CURRENT_LANGUAGE, CONFIRM_WORDS["en"])
    cancels  = CANCEL_WORDS.get(CURRENT_LANGUAGE, CANCEL_WORDS["en"])

    if any(w in decision_text for w in confirms):
        print("   ✅ Voice confirmed — executing")
        return "execute", None
    elif any(w in decision_text for w in cancels):
        print("   🗑️  Voice cancelled — discarding")
        return "discard", None
    else:
        print(f"   ⚠️  Unclear ('{decision_text}') — defaulting to execute")
        return "execute", None


def _hitl_edit_args(pending_tool_calls: list) -> tuple[str, str | None]:
    if not pending_tool_calls:
        return "execute", None

    tc = pending_tool_calls[0]
    try:
        current = json.loads(tc.get("arguments", "{}"))
    except (json.JSONDecodeError, TypeError):
        current = {}

    print(f"\n✏️  Editing arguments for: {tc['name']}")
    print(f"   Current args: {json.dumps(current, indent=2)}")
    print("   Enter key=value pairs. Press ENTER on empty line when done.\n")

    new_args = dict(current)
    while True:
        try:
            line = input("   > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if "=" not in line:
            print("   ⚠️  Format must be key=value. Try again.")
            continue
        key, _, raw_val = line.partition("=")
        key = key.strip(); raw_val = raw_val.strip()
        if raw_val.lower() in ("true", "false"):
            value = raw_val.lower() == "true"
        elif raw_val.isdigit():
            value = int(raw_val)
        else:
            try:
                value = float(raw_val)
            except ValueError:
                value = raw_val
        new_args[key] = value
        print(f"   ✓ Set {key} = {value!r}")

    edited_json = json.dumps(new_args)
    print(f"\n   Final args: {json.dumps(new_args, indent=2)}")
    print("✅ Executing with edited arguments...\n")
    return "edit", edited_json


# ============================================================
# MODULE 1 — WAKE WORD DETECTOR
# ============================================================

class WakeWordDetector:
    def __init__(self, on_detected, audio_interface: pyaudio.PyAudio):
        self.on_detected     = on_detected
        self._audio          = audio_interface
        self._stop_event     = Event()
        self._paused         = False
        self._cooldown_until = 0.0
        self._thread: Thread = None
        self._stream         = None

    def start(self):
        self._thread = Thread(target=self._run, daemon=True, name="WakeWordDetector")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def pause(self):
        self._paused = True

    def resume(self, extra_cooldown: float = COOLDOWN_TIME):
        self._cooldown_until = time.time() + extra_cooldown
        self._paused = False

    def _is_in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def _run(self):
        print("📥 Downloading openWakeWord base models...")
        try:
            openwakeword.utils.download_models()
        except Exception as e:
            print(f"   Note: {e}")

        print(f"\n🔧 Loading wake-word model: {MODEL_PATH}")
        model = Model(
            wakeword_models=[MODEL_PATH],
            vad_threshold=0,
            inference_framework="onnx",
        )
        print(f"✓ Model ready: {list(model.models.keys())}")
        print(f"✓ Model input size: {model.model_inputs}")

        try:
            instr = pygame.mixer.Sound(INSTRUCTION_SOUND)
            instr.play()
            while pygame.mixer.get_busy():
                time.sleep(0.05)
            print("✓ Instruction sound played — now listening for wake word")
        except Exception as e:
            print(f"[InstructionSound] Error: {e}")

        self._stream = self._audio.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK_SIZE)
        print("✓ Wake-word audio stream opened\n")

        frame_count     = 0
        detection_count = 0
        score_history   = deque(maxlen=10)
        max_score_seen  = 0.0

        try:
            while not self._stop_event.is_set():
                raw   = self._stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frame = np.frombuffer(raw, dtype=np.int16)

                if self._paused or self._is_in_cooldown():
                    if self._is_in_cooldown() and frame_count % 50 == 0:
                        remaining = self._cooldown_until - time.time()
                        print(f"Cooldown: {remaining:.1f}s remaining", end="\r")
                    frame_count += 1
                    continue

                prediction = model.predict(frame)

                for key, score in prediction.items():
                    score_history.append(score)
                    smoothed = sum(score_history) / len(score_history)

                    if score > max_score_seen:
                        max_score_seen = score
                        print(f"\n📈 New max: raw={score:.5f} | smooth={smoothed:.5f}")

                    if frame_count % 50 == 0:
                        amp = float(np.abs(frame).mean())
                        print(
                            f"score={score:.5f} | smooth={smoothed:.5f} | "
                            f"max={max_score_seen:.5f} | amp={amp:.0f}",
                            end="\r")

                    if smoothed >= DETECTION_THRESHOLD:
                        detection_count += 1
                        print(f"\n🎯 WAKE WORD DETECTED! "
                              f"(smooth={smoothed:.5f} raw={score:.5f} #{detection_count})")

                        model.reset()
                        score_history.clear()
                        max_score_seen = 0.0
                        self._cooldown_until = time.time() + COOLDOWN_TIME
                        self.pause()

                        self._stream.stop_stream()
                        self._stream.close()
                        self._stream = None

                        self.on_detected()

                        self._stream = self._audio.open(
                            format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK_SIZE)
                        print("\n👂 Wake-word stream reopened — listening...\n")
                        break

                frame_count += 1

        except Exception as e:
            if not self._stop_event.is_set():
                print(f"\n✗ WakeWordDetector error: {e}")
                import traceback; traceback.print_exc()
        finally:
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
            print("WakeWordDetector stopped.")


# ============================================================
# MODULE 2 — VOICE RECORDER
# ============================================================

class VoiceRecorder:
    def __init__(self, audio_interface: pyaudio.PyAudio):
        self._audio = audio_interface

    def record(self) -> bytes | None:
        print("\n🎙️  Recording started (own stream)...")
        frames        = []
        max_chunks    = int(RECORDING_DURATION * RATE / CHUNK_SIZE)
        min_chunks    = int(MIN_RECORDING_TIME * RATE / CHUNK_SIZE)
        silent_frames = 0
        voice_seen    = False
        speech_chunks = 0
        recorded      = 0

        stream = self._audio.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK_SIZE)

        for _ in range(3):
            stream.read(CHUNK_SIZE, exception_on_overflow=False)

        try:
            for _ in range(max_chunks):
                raw  = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(raw)
                recorded += 1

                arr        = np.frombuffer(raw, dtype=np.int16)
                has_speech = _check_vad_buffer(arr, num_frames=2)

                if has_speech:
                    speech_chunks += 1
                    if not voice_seen:
                        voice_seen = True
                        print("   ✓ Voice detected, recording...")
                    silent_frames = 0
                else:
                    if recorded >= min_chunks:
                        silent_frames += 1

                if recorded >= min_chunks and voice_seen:
                    if silent_frames > 0 and silent_frames % 5 == 0:
                        secs = silent_frames * CHUNK_SIZE / RATE
                        print(f"   Silence: {secs:.1f}s / {SILENCE_DURATION:.1f}s", end="\r")
                    if silent_frames >= VAD_SILENCE_CHUNKS:
                        print("\n   ✓ Finished (silence detected)")
                        break

                if recorded % 10 == 0:
                    elapsed = recorded * CHUNK_SIZE / RATE
                    print(f"   Recording… {elapsed:.1f}s | speech chunks: {speech_chunks}", end="\r")
            else:
                print("\n   ✓ Finished (max duration)")

        except Exception as e:
            print(f"\n✗ Recording error: {e}")
        finally:
            stream.stop_stream()
            stream.close()

        audio_bytes             = b"".join(frames)
        is_valid, reason, stats = _validate_recording(audio_bytes)

        print(f"\n📊 Recording stats:")
        print(f"   Duration  : {stats.get('duration', 0):.2f}s")
        print(f"   Speech    : {stats.get('speech_percentage', 0):.1f}%  "
              f"({stats.get('speech_frames', 0)} frames)")
        print(f"   Amplitude : {stats.get('mean_amplitude', 0):.0f} avg  "
              f"/ {stats.get('max_amplitude', 0):.0f} max")

        if is_valid:
            print(f"   ✅ {reason}")
            return audio_bytes
        else:
            print(f"   ❌ {reason} — discarding, not sending to agent")
            return None


# ============================================================
# MODULE 3 — AGENT CLIENT
# ============================================================

class AgentClient:

    def __init__(self, recorder):
        self._recorder           = recorder
        self._last_response_text = ""
        self._in_conversation    = False
        self._ws                 = None
        self._thread_id          = str(uuid.uuid4())
        print(f"[AgentClient] Permanent thread_id: {self._thread_id}")

    # ------------------------------------------------------------------ #

    def send(self, audio_bytes: bytes):
        """Entry point — runs ONE event loop for the entire conversation."""
        asyncio.run(self._conversation_loop(audio_bytes))

    # ------------------------------------------------------------------ #

    async def _conversation_loop(self, audio_bytes: bytes):
        """Single event loop — WebSocket stays open across all follow-up turns."""
        self._in_conversation = True

        # Connect ONCE for the entire conversation
        await self._ensure_connected()

        while self._in_conversation:
            b64 = _audio_to_base64_wav(audio_bytes)

            # Wait-moment sound
            try:
                wait_snd = pygame.mixer.Sound(WAIT_MOMENT_SOUND)
                pygame.mixer.Channel(0).play(wait_snd)
                while pygame.mixer.Channel(0).get_busy():
                    await asyncio.sleep(0.05)
                print("✓ Wait-moment sound finished")
            except Exception as e:
                print(f"[WaitSound] Error: {e}")

            _thinking_player.start()
            try:
                await self._exchange_on_open_ws(b64)
            finally:
                _thinking_player.stop()

            last = self._last_response_text.strip()
            print(f"   [ConvLoop] Last assistant text: {last!r}")

            if not _needs_followup(last):
                print("✅ Conversation complete — returning to wake-word listening")
                self._in_conversation = False
                break

            # Agent expects a reply — listen without wake word
            print("💬 Agent needs a follow-up — listening (no wake word needed)…")
            try:
                beep    = pygame.mixer.Sound(HITL_REVIEW_SOUND)
                channel = pygame.mixer.Channel(2)
                channel.play(beep)
                while channel.get_busy():
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[ConvBeep] Error: {e}")

            # Record in thread executor so event loop stays alive
            loop        = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(None, self._recorder.record)

            if not audio_bytes:
                print("   ⚠️  No audio — ending conversation")
                self._in_conversation = False
                break

            arr     = np.frombuffer(audio_bytes, dtype=np.int16)
            max_amp = float(np.abs(arr).max())
            avg_amp = float(np.abs(arr).mean())
            if max_amp / (avg_amp + 1) < 2.5:
                print("   ⚠️  Likely noise — ending conversation")
                self._in_conversation = False
                break

        self._in_conversation = False

    # ------------------------------------------------------------------ #

    async def _ensure_connected(self):
        """Open connection if not already open, send permanent thread_id."""
        if self._ws is None or self._ws.closed:
            self._ws = await websockets.connect(
                WEBSOCKET_URI,
                max_size=50 * 1024 * 1024,
                ping_interval=30,
                ping_timeout=120,
            )
            print("[WS] Connected to server")
            # Receive session_init (ignore server's generated thread_id)
            await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            # Send our permanent thread_id
            await self._ws.send(json.dumps({
                "type":      "session_resume",
                "thread_id": self._thread_id
            }))
            print(f"[WS] Using permanent thread_id: {self._thread_id}")

    async def _disconnect(self):
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None
        # Never reset self._thread_id

    # ------------------------------------------------------------------ #

    async def _exchange_on_open_ws(self, b64_audio: str):
        """Send audio on the already-open connection — no reconnect."""
        try:
            await self._ws.send(json.dumps({
                "type":     "audio",
                "audio":    b64_audio,
                "language": CURRENT_LANGUAGE,
            }))
            print("📤 Sent audio to agent…")
            await self._receive_loop(self._ws)

        except websockets.exceptions.ConnectionClosed:
            print("✗ Connection closed mid-conversation")
            self._ws = None
            self._in_conversation = False
        except Exception as e:
            print(f"✗ Agent error: {e}")
            self._ws = None
            self._in_conversation = False

    # ------------------------------------------------------------------ #

    async def _receive_loop(self, ws):
        while True:
            try:
                raw  = await asyncio.wait_for(ws.recv(), timeout=60.0)
                data = json.loads(raw)
                kind = data.get("type")
                print(f"📥 Received: {kind}")

                if kind == "transcription":
                    text = data["text"].lower().strip()
                    print(f"   You      : {data['text']}")
                    cleaned = text.strip(".,!?;:- ").strip()
                    if not cleaned or len(cleaned) < 2:
                        print("   ⚠️  Empty transcription — ignoring")
                    if "switch to urdu" in text or "urdu mein bolo" in text:
                        switch_language("ur")
                        await ws.send(json.dumps({"type": "switch_language", "language": "ur"}))
                    elif "switch to hindi" in text or "hindi mein bolo" in text:
                        switch_language("hi")
                        await ws.send(json.dumps({"type": "switch_language", "language": "hi"}))
                    elif "switch to english" in text or "speak english" in text:
                        switch_language("en")
                        await ws.send(json.dumps({"type": "switch_language", "language": "en"}))

                elif kind == "response":
                    rd       = data["data"]
                    msg_text = rd.get("message", "")
                    self._last_response_text = msg_text
                    print(f"   Assistant: {msg_text}")

                    is_api_error = (
                        "Error code:" in msg_text
                        or "tool_use_failed" in msg_text
                        or "failed_generation" in msg_text
                        or (isinstance(msg_text, str) and msg_text.strip().startswith("Error"))
                    )
                    if is_api_error:
                        clean_error = (
                            "I'm sorry, I couldn't process that request. "
                            "Please try again with valid information."
                        )
                        print("   ⚠️  API error detected — speaking clean message instead")
                        error_audio = text_to_speech(clean_error)
                        if error_audio:
                            _thinking_player.stop()
                            err_sound = pygame.mixer.Sound(io.BytesIO(error_audio))
                            err_sound.play()
                            while pygame.mixer.get_busy():
                                await asyncio.sleep(0.1)

                    if rd.get("function_called"):
                        print(f"   Function : {rd['function_called']}")

                    if rd.get("function_called") == "check_result":
                        results = rd.get("function_results", [])
                        for r in results:
                            d = r.get("result", {}).get("data", {})
                            if d:
                                print("\n" + "━" * 40)
                                print("📋  RESULT")
                                print("━" * 40)
                                print(f"  Roll No : {d.get('roll_number', 'N/A')}")
                                print(f"  Name    : {d.get('name', 'N/A')}")
                                print(f"  Marks   : {d.get('marks', 'N/A')}")
                                print(f"  Grade   : {d.get('grade', 'N/A')}")
                                print("━" * 40 + "\n")

                elif kind == "pending_approval":
                    if HITL_ENABLED:
                        await self._handle_hitl(ws, data["data"])
                    else:
                        await ws.send(json.dumps({"type": "hitl_decision", "decision": "execute"}))

                elif kind == "audio_response":
                    _thinking_player.stop()
                    print("🔊 Playing audio response…")
                    # Play in executor so event loop stays responsive
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._play, data["audio"])
                    break

                elif kind == "reminder":
                    print(f"\n⏰ {data['data']['message']}")

                elif kind == "error":
                    print(f"   ✗ Agent error: {data['message']}")
                    break

            except asyncio.TimeoutError:
                print("⏱️  Response timeout")
                break
            except json.JSONDecodeError as e:
                print(f"✗ JSON error: {e}")
                break

    async def _handle_hitl(self, ws, payload: dict):
        pending_calls = payload.get("pending_tool_calls", [])
        thread_id     = payload.get("thread_id", "default")

        _thinking_player.stop()

        loop                    = asyncio.get_event_loop()
        decision, edited_args   = await loop.run_in_executor(
            None, _hitl_review_voice, pending_calls, self._recorder
        )

        msg = {
            "type":      "hitl_decision",
            "thread_id": thread_id,
            "decision":  decision,
        }
        if edited_args is not None:
            msg["edited_arguments"] = edited_args

        await ws.send(json.dumps(msg))
        print(f"📤 Decision sent: {decision}")
        _thinking_player.start()

    @staticmethod
    def _play(b64_audio: str):
        try:
            data  = base64.b64decode(b64_audio)
            sound = pygame.mixer.Sound(io.BytesIO(data))
            sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.1)
            time.sleep(0.3)
            print("✓ Audio response finished")
        except Exception as e:
            print(f"✗ Playback error: {e}")



# ============================================================
# MODULE 4 — ORCHESTRATOR
# ============================================================

class AssistantOrchestrator:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        self._audio    = pyaudio.PyAudio()
        self._detector = WakeWordDetector(
            on_detected=self._on_wake_word,
            audio_interface=self._audio)
        self._recorder = VoiceRecorder(self._audio)
        self._agent    = AgentClient(self._recorder)

    def run(self):
        self._print_banner()
        try:
            self._detector.start()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down…")
        finally:
            self._detector.stop()
            self._audio.terminate()
            pygame.mixer.quit()
            print("✓ Voice assistant stopped. Goodbye! 👋\n")

    def _on_wake_word(self):
        self._play_ding()
        while pygame.mixer.get_busy():
            time.sleep(0.05)
        time.sleep(0.2)

        self._play_beep()
        audio_bytes = self._recorder.record()

        if not audio_bytes:
            print("   ⚠️  No valid speech — ending conversation")
            self._detector.resume(extra_cooldown=COOLDOWN_TIME)
            return

        arr     = np.frombuffer(audio_bytes, dtype=np.int16)
        max_amp = float(np.abs(arr).max())
        avg_amp = float(np.abs(arr).mean())
        ratio   = max_amp / (avg_amp + 1)
        if ratio < 2.5:
            print(f"   ⚠️  Likely noise not speech (ratio={ratio:.1f}) — skipping")
            self._detector.resume(extra_cooldown=COOLDOWN_TIME)
            return

        self._agent.send(audio_bytes)

        self._detector.resume(extra_cooldown=COOLDOWN_TIME)

    @staticmethod
    def _play_ding():
        try:
            pygame.mixer.music.load(RESPONSE_AUDIO_PATH)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("✓ Ding played")
        except Exception as e:
            print(f"✗ Ding playback error: {e}")

    @staticmethod
    def _play_beep():
        try:
            beep    = pygame.mixer.Sound(HITL_REVIEW_SOUND)
            channel = pygame.mixer.Channel(2)
            channel.play(beep)
            while channel.get_busy():
                time.sleep(0.05)
            time.sleep(0.1)
            print("✓ Beep played — recording now")
        except Exception as e:
            print(f"[BEEP] Error: {e}")

    @staticmethod
    def _print_banner():
        print("\n" + "=" * 70)
        print("🤖  AI VOICE ASSISTANT — SEPARATED ARCHITECTURE")
        print("=" * 70)
        print(f"  Wake-word model : {MODEL_PATH}")
        print(f"  Threshold       : {DETECTION_THRESHOLD}")
        print(f"  VAD aggressiveness: {VAD_MODE}")
        print(f"  Agent URI       : {WEBSOCKET_URI}")
        print(f"  HITL mode       : {'ON' if HITL_ENABLED else 'OFF'}")
        print("=" * 70)
        print("\n👂  Listening for wake word…  (Ctrl-C to stop)\n")

        print("Available audio devices:")
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate()
        print()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    try:
        AssistantOrchestrator().run()
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()