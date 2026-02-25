# OpenAvatarChat – Changes Summary

## What We Changed

1. **ASR: SenseVoice → Whisper** – Replaced SenseVoice (hallucinated on noise) with faster-whisper (base, int8, CUDA, English-only). Eliminated phantom transcriptions.

2. **LLM: Ollama llama3.2:1b** – Local LLM at `localhost:11434`, no cloud dependency. Short English system prompt.

3. **VAD Tuning** – Lowered threshold (0.5→0.45), faster start detection (2048→1536 samples), more padding (512→1200). Faster barge-in, fewer cut-off words.

4. **Barge-In System (Major Fix)** – The interrupt pipeline was completely broken:
   - `emit_signal()` was a no-op – signals never reached the backend. Fixed by wiring callback to ChatSession.
   - TTS had no cancellation – now checks interrupt flag during synthesis and abandons immediately.
   - Avatar queues were never flushed – now clears audio/video on interrupt.
   - All handler queues now drain on interrupt, VAD re-enables instantly.
   - **Result:** User can now interrupt mid-sentence and the entire pipeline stops.

5. **Frontend UI** – Fixed avatar video rendering (`v-if`→`v-show` WebRTC fix), styled buttons (purple gradient), polished chat bubbles.

6. **TTS & Avatar Config** – Edge TTS (`en-US-AriaNeural`, 24kHz), LiteAvatar (GPU, 25fps, faster cleanup).

## Models

| Component | Model |
|-----------|-------|
| ASR | faster-whisper base (int8, CUDA) |
| LLM | llama3.2:1b (Ollama, local) |
| TTS | Edge TTS en-US-AriaNeural |
| VAD | Silero VAD (ONNX, tuned) |
| Avatar | LiteAvatar (GPU, 25fps) |

## Files Modified

- `config/chat_with_openai_compatible_edge_tts.yaml` – ASR, VAD, LLM, Avatar config
- `src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py` – Barge-in & interrupt lifecycle
- `src/handlers/tts/edgetts/tts_handler_edgetts.py` – Interrupt-aware synthesis
- `src/handlers/avatar/liteavatar/avatar_handler_liteavatar.py` – Queue flushing on interrupt
- `src/handlers/client/rtc_client/client_handler_rtc.py` – Fixed emit_signal routing
- `src/chat_engine/common/client_handler_base.py` – Signal callback wiring
- `src/chat_engine/core/chat_session.py` – Pipeline flush on interrupt
- `src/handlers/vad/silerovad/vad_handler_silero.py` – Barge-in detection
- `src/service/rtc_service/rtc_stream.py` – stop_chat cleanup
- `frontend/src/views/VideoChat/index.vue` – v-if→v-show fix
- `frontend/src/views/VideoChat/index.less` – z-index, styling
- `frontend/src/components/ActionGroup.vue` – Button styling
- `frontend/src/components/ChatMessage.vue` – Message styling
- `frontend/src/components/ChatInput.vue` – Input styling
| `frontend/src/components/ChatRecords.vue` | Message animation |
