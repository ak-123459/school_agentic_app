"""
server.py — WebSocket server with Human-in-the-Loop support
============================================================
Adds three new message types on top of the original server:

  Client → Server:
    { type: "hitl_decision",
      thread_id: "...",
      decision: "continue" | "discard" | "edit" | "execute",
      edited_arguments: "..."  (optional, JSON string)
    }

  Server → Client  (when an action needs approval):
    { type: "pending_approval",
      data: {
        message: "...",
        pending_tool_calls: [...],
        thread_id: "..."
      }
    }

  Server → Client  (after resume):
    { type: "response", data: { ... } }   (same as before)

All original message types (audio, text, reminder, audio_response) are
fully preserved.
"""

import asyncio
import json
import base64
import uuid

import websockets

from reasoning_engine.assistant import AIVoiceAssistant
from reasoning_engine.tts.tts import text_to_speech
from datetime import datetime
import yaml



# Load the YAML file
with open("configs/assist_config.yaml", "r", encoding="utf-8") as f:
    assist_config = yaml.safe_load(f)

# ── Load server config ────────────────────────────────────────
with open("configs/server_config.yaml", "r", encoding="utf-8") as f:
    server_config = yaml.safe_load(f)


CURRENT_LANGUAGE = assist_config['CURRENT_LANGUAGE']
LANGUAGE_NAMES = assist_config['LANGUAGE_NAMES']
SYSTEM_PROMPT = assist_config['system_prompt']['uptal_voice_assistant']

_SRV  = server_config["server"]
_HITL = server_config["hitl"]
_CONV = server_config["conversation"]
_TTS  = server_config["tts"]
_ERR  = server_config["error_filters"]
HITL_MODE = _HITL["mode"]

# 1. _send_response_with_tts — replace hardcoded 300
MAX_CHARS = _TTS["max_chars"]


# ── Global assistant instance (shared across all connections) ───────────────
assistant = AIVoiceAssistant()



# ── HITL mode: set to "review" to pause before every tool call ─────────────
#    set to "auto" to run tool calls without interruption (original behaviour)
HITL_MODE = "review"   # change to "auto" to disable human-in-the-loop


# ============================================================
# BACKGROUND: REMINDER CHECKER (unchanged)
# ============================================================


async def _send_response_with_tts(websocket, response: dict):

    message = response.get("message") or ""
    status  = response.get("status", "success")

    # Fallback to tool result if LLM returned None
    if not message and response.get("function_results"):
        for r in response.get("function_results", []):
            result_data = r.get("result", {})
            message = result_data.get("message", "")
            if message:
                break

    # ── Never speak raw error strings to the user ─────────────────
    if status == "error":
        err = message.lower()
        if any(x in err for x in ("error code", "failed_generation", "tool_use_failed",
                                  "badrequest", "http/1.1 4", "groq", "'type'")):
            message = "Sorry, I couldn't complete that request. Please try again."


    # ── Truncate long responses for voice ───────────────────────────
    MAX_CHARS = 300
    if len(message) > MAX_CHARS:
        message = message[:MAX_CHARS].rsplit(" ", 1)[0] + "..."

    slim_response = {
        "status":          response.get("status"),
        "message":         message,
        "function_called": response.get("function_called"),
    }

    await websocket.send(json.dumps({"type": "response", "data": slim_response}))
    print(f"[SEND] AI response sent | status={response.get('status')}")

    if message:
        print("[INFO] Generating TTS audio")
        audio_content = text_to_speech(message)
        if audio_content:
            audio_base64 = base64.b64encode(audio_content).decode("utf-8")
            await websocket.send(json.dumps({
                "type": "audio_response",
                "audio": audio_base64
            }))
            print("[SEND] TTS audio sent")


async def _process_text(
    websocket,
    text: str,
    client_conversation: list,
    thread_id: str
):
    """
    Core processing pipeline for a user text message.
    Handles both "success" and "pending" responses.
    """
    response = await assistant.process_command(
        text,
        client_conversation,
        thread_id=thread_id,
        hitl_mode=HITL_MODE
    )

    print(f"[AI RESPONSE] status={response['status']} | {str(response.get('message', ''))[:80]}")

    # ── Update client-side conversation history ──────────────────────────
    client_conversation.append({"role": "user", "content": text})

    if response.get("conversation_update"):

        updates = response["conversation_update"]
        if isinstance(updates, list):
            client_conversation.extend(updates)
        else:
            client_conversation.append(updates)

    if len(client_conversation) > 7:
        client_conversation = [client_conversation[0]] + client_conversation[-6:]
        print("[INFO] Truncated conversation history")

    # ── PENDING: tool call needs approval ────────────────────────────────
    if response["status"] == "pending":
        await websocket.send(json.dumps({
            "type": "pending_approval",
            "data": {
                "message":            response["message"],
                "pending_tool_calls": response.get("pending_tool_calls", []),
                "thread_id":          response["thread_id"],
            }
        }))
        print(f"[HITL] Paused — awaiting human decision for thread {response['thread_id']}")
        # Do NOT send TTS here; wait for resume
        return

    # ── SUCCESS or ERROR: send immediately ───────────────────────────────
    await _send_response_with_tts(websocket, response)


# ============================================================
# MAIN CONNECTION HANDLER
# ============================================================

async def handle_client(websocket, path):

    global CURRENT_LANGUAGE  # ← add this
    thread_id = str(uuid.uuid4())
    print(f"[CONNECT] Client connected: {websocket.remote_address}")
    print(f"[INFO] Assigned thread_id: {thread_id}")
    current_language = CURRENT_LANGUAGE  # ← local copy per connection


    # ── Send thread_id to client immediately ──────────────────────────────
    await websocket.send(json.dumps({
        "type": "session_init",
        "thread_id": thread_id,
    }))


    # Check if client wants to resume an existing thread
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        data = json.loads(raw)
        if data.get("type") == "session_resume":
            thread_id = data["thread_id"]  # ← use client's permanent ID
            print(f"[INFO] Resuming thread_id: {thread_id}")
        else:
            # Not a resume — process as normal first message
            # re-queue it by handling inline (see note below)
            pass
    except asyncio.TimeoutError:
        pass  # new client, keep generated thread_id

    print(f"[INFO] thread_id: {thread_id}")


    client_conversation = [{
        "role": "system",
        "content": SYSTEM_PROMPT.format(
            datetime=datetime.now().strftime("%A, %B %d %Y %I:%M %p"),
            language=LANGUAGE_NAMES[current_language]
        )
    }]



    try:
        async for message in websocket:
            try:
                print(f"[RECEIVED] Raw message: {message[:100]}...")
                data = json.loads(message)
                msg_type = data.get("type")
                print(f"[PARSE] type={msg_type}")

                # ── AUDIO MESSAGE ────────────────────────────────────────
                if msg_type == "audio":

                    lang = data.get("language", "en")  # ← read language from message
                    text = assistant.process_audio(data["audio"], language=lang)  # ← pass it
                    if text and not text.startswith("[ERROR]") and not text.startswith("[WARN]"):
                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": text
                        }))
                        await _process_text(websocket, text, client_conversation, thread_id)
                    else:
                        print(f"[WARN] Audio processing failed: {text}")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Could not process audio. Please try again."
                        }))

                # ── TEXT MESSAGE ─────────────────────────────────────────
                elif msg_type == "text":
                    print("[INFO] Text message received")
                    text = data["text"]
                    print(f"[USER MESSAGE] {text}")
                    await _process_text(websocket, text, client_conversation, thread_id)

                elif msg_type == "switch_language":

                    new_lang = data.get("language", "en")
                    CURRENT_LANGUAGE = new_lang

                    client_conversation[0] = {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(
                            datetime=datetime.now().strftime("%A, %B %d %Y %I:%M %p"),
                            language=LANGUAGE_NAMES[current_language]
                        )
                    }
                    print(f"[LANG] Switched to: {LANGUAGE_NAMES[current_language]}")
                    await websocket.send(json.dumps({
                        "type": "language_switched",
                        "language": new_lang,
                        "name": LANGUAGE_NAMES[current_language]
                    }))

                # ── HUMAN-IN-THE-LOOP DECISION ────────────────────────────
                elif msg_type == "hitl_decision":
                    """
                    Expected payload:
                    {
                      "type": "hitl_decision",
                      "thread_id": "...",
                      "decision": "continue" | "discard" | "edit" | "execute",
                      "edited_arguments": "..."   (optional, only for "edit")
                    }
                    """
                    decision_thread_id  = data.get("thread_id", thread_id)
                    decision            = data.get("decision", "execute")
                    edited_arguments    = data.get("edited_arguments")

                    print(f"[HITL] Decision received: {decision} for thread {decision_thread_id}")

                    response = await assistant.resume_command(
                        thread_id=decision_thread_id,
                        decision=decision,
                        edited_arguments=edited_arguments
                    )

                    # Update conversation history after resume
                    if response.get("conversation_update"):
                        updates = response["conversation_update"]

                        if isinstance(updates, list):

                            client_conversation.extend(updates)
                        else:

                            client_conversation.append(updates)


                    # ✅ Mutates the original list — caller sees the change
                    if len(client_conversation) > _CONV["max_history"]:
                        trimmed = [client_conversation[0]] + client_conversation[_CONV["keep_recent"]:]
                        client_conversation.clear()
                        client_conversation.extend(trimmed)


                    await _send_response_with_tts(websocket, response)

                # ── GET THREAD HISTORY ────────────────────────────────────
                elif msg_type == "get_history":
                    req_thread = data.get("thread_id", thread_id)
                    history = assistant.get_thread_history(req_thread)
                    await websocket.send(json.dumps({
                        "type": "history",
                        "thread_id": req_thread,
                        "data": history
                    }))

                # ✅ Fix — format it the same way as initial setup
                elif msg_type == "clear_thread":
                    req_thread = data.get("thread_id", thread_id)
                    success = assistant.clear_thread(req_thread)

                    client_conversation = [{
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(
                            datetime=datetime.now().strftime("%A, %B %d %Y %I:%M %p"),
                            language=LANGUAGE_NAMES[current_language]
                        )
                    }]

                    await websocket.send(json.dumps({
                        "type": "thread_cleared",
                        "thread_id": req_thread,
                        "success": success
                    }))


                    print(f"[INFO] Thread {req_thread} cleared")

                else:
                    print(f"[WARN] Unknown message type: {msg_type}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    }))

            except json.JSONDecodeError:
                print("[ERROR] Invalid JSON format")
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON format"}))
            except Exception as e:
                print(f"[ERROR] Processing message failed: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        print(f"[DISCONNECT] Client disconnected: {websocket.remote_address}")
    finally:
        pass










# ============================================================
# START SERVER
# ============================================================

async def start_server():
    import websockets

    print("[SERVER] Starting WebSocket server...")
    print(f"[SERVER] HITL mode: {HITL_MODE}")
    print("[SERVER] Host: localhost | Port: 8765")



    # ✅ Fix — add ping_interval and ping_timeout
    async with websockets.serve(
            handle_client,
            _SRV["host"],
            _SRV["port"],
            max_size=_SRV["max_size_mb"] * 1024 * 1024,
            ping_interval=_SRV["ping_interval_seconds"],
            ping_timeout=_SRV["ping_timeout_seconds"],
    ):
        print("[SERVER] ✓ WebSocket server running!\n")
        await asyncio.Future()