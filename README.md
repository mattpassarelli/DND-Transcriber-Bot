# D&D Scribe Bot

A locally-hosted Discord bot that joins your voice channel, transcribes everyone in real time using Whisper, and generates structured D&D session notes using a local LLM via Ollama. No audio or transcript data ever leaves your machine.

Built on top of [Discord-Realtime-STT-Bot](https://github.com/Leehyunbin0131/Discord-Realtime-STT-Bot) with D&D-specific summarization, session management commands, and crash-safe rolling transcript logging.

---

## How It Works

```
Discord Voice → discord-ext-voice-recv → STTAudioSink
    → Resample (48kHz stereo → 16kHz mono)
    → IPC Queue → STT Subprocess
        → Silero VAD (speech detection)
        → faster-whisper (speech-to-text)
    → Result Queue → Bot Process
        → Rolling transcript file (written live)
        → !summary / !leave → Ollama (LLM summarization)
            → Structured session notes posted to Discord
            → Markdown file saved to disk
```

The bot runs two processes:

- **Bot process** — handles Discord connection, voice receive, commands, and result polling
- **STT subprocess** — runs Silero VAD and faster-whisper in isolation so GPU inference never blocks the Discord event loop

Audio flows from Discord through a per-user resampler, into a multiprocessing queue, through voice activity detection, and into Whisper. Transcribed lines are written to both an in-memory list and a rolling file on disk as they arrive. At the end of a session, Ollama summarizes the full transcript into structured D&D notes.

---

## Requirements

### Hardware
- NVIDIA GPU with 12GB+ VRAM recommended
- Tested on RTX 5070 Ti (16GB) — Blackwell architecture requires special setup (see below)

### Software
- Windows 10/11 or Linux
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com) (local LLM server)
- [FFmpeg](https://ffmpeg.org)
- NVIDIA driver 572.16+ (for RTX 5000 series / Blackwell)

---

## Setup

### 1. Install Prerequisites

**Ollama** — download and install from [ollama.com](https://ollama.com), then pull a model:
```
ollama pull qwen3:14b
```

**FFmpeg** — install via winget on Windows:
```
winget install ffmpeg
```

**uv** — install the package manager:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Configure

```bash
git clone https://github.com/yourusername/dnd-scribe.git
cd dnd-scribe
cp .env.example .env
```

Edit `.env` with your settings — at minimum, set your `DISCORD_TOKEN`.

### 3. Install Dependencies

```bash
uv sync
```

**PyTorch — CUDA setup (important)**

PyTorch must be installed from the PyTorch index, not PyPI, to get CUDA support. The right CUDA version depends on your GPU:

| GPU Generation | CUDA Version | Command |
|---|---|---|
| RTX 5000 series (Blackwell) | cu130 | See below |
| RTX 3000/4000 series | cu128 | `--index-url https://download.pytorch.org/whl/cu128` |

For RTX 5000 series (Blackwell):
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Verify PyTorch can see your GPU:
```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name.

### 4. Create a Discord Bot

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Click **New Application**
3. Go to the **Bot** tab → click **Reset Token** → copy the token into your `.env`
4. Under **Privileged Gateway Intents**, enable all three:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
5. Go to **OAuth2 → URL Generator** → check `bot` scope → check these permissions:
   - Read Messages / View Channels
   - Send Messages
   - Connect
   - Speak
6. Copy the generated URL, open it in your browser, and invite the bot to your server

### 5. Run

```bash
uv run main.py
```

You should see:
```
INFO  │ STT process spawned (PID ...)
INFO  │ Silero VAD loaded successfully.
INFO  │ Faster-Whisper loaded on CUDA.
INFO  │ STT Processor ready
INFO  │ Logged in as D&D Scribe#1234
```

---

## Configuration

All settings are in `.env`. Copy `.env.example` to get started.

| Variable | Default | Description |
|---|---|---|
| `DISCORD_TOKEN` | — | **Required.** Your bot token from the Developer Portal |
| `COMMAND_PREFIX` | `!` | Prefix for bot commands |
| `STT_MODEL_ID` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Whisper model to use |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `int8_float16` | Use `int8_float16` for Blackwell GPUs, `float16` for others |
| `STT_LANGUAGE` | `en` | Language code for Whisper transcription |
| `STT_BEAM_SIZE` | `1` | Higher = more accurate but slower |
| `STT_MIN_AUDIO_BYTES` | `8000` | Minimum audio chunk size before transcription (~0.25s) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model to use for summarization |
| `OLLAMA_TIMEOUT` | `180` | Seconds to wait for Ollama response |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Choosing an Ollama Model

For summarization quality vs VRAM fit on a single GPU:

| Model | VRAM (Q4) | Notes |
|---|---|---|
| `llama3.1:8b` | ~5GB | Fast, decent quality |
| `qwen3:14b` | ~9GB | Recommended — better instruction following |
| `mistral-small:22b` | ~14GB | Strong quality, tight fit on 16GB alongside Whisper |

For Qwen3 models, add `/no_think` at the top of the prompts in `bot/summarizer.py` to disable the chain-of-thought reasoning mode, which is unnecessary for structured note generation.

### Customizing the D&D Prompt

Open `bot/summarizer.py` and edit the `CAMPAIGN_CONTEXT` variable at the top:

```python
CAMPAIGN_CONTEXT = "The party: Aldric (human paladin), Zessa (tiefling warlock), Pip (halfling rogue). Campaign: Curse of Strahd, currently in Barovia."
```

The more context you add here, the better the notes — the model has no memory of previous sessions, so anything you bake into the prompt improves how well it identifies characters and plot threads.

---

## Commands

| Command | Description |
|---|---|
| `!join` | Join your current voice channel and start transcribing |
| `!leave` | Flush audio, dump raw transcript, post session notes, and disconnect |
| `!summary` | Generate and post session notes without stopping the recording |
| `!dumptranscript` | Save the raw transcript to a file immediately, no summarization |
| `!clearnotes` | Wipe the in-memory transcript and start fresh from this point |
| `!status` | Show how long the session has been running and how many lines are captured |

---

## Output Files

All files are saved to the `sessions/` directory in your project folder.

| File | When created | Contents |
|---|---|---|
| `rolling_<timestamp>.txt` | On `!join`, written continuously | Every transcription line as it arrives — crash-safe |
| `transcript_<timestamp>.txt` | On `!leave`, `!summary`, `!dumptranscript` | Full raw transcript dump |
| `session_<timestamp>_final.md` | On `!leave` | Structured session notes from Ollama |
| `session_<timestamp>_checkpoint.md` | On `!summary` | Mid-session notes snapshot |

The rolling file is the most important for crash safety — it's written and flushed to disk after every transcription line, so a Ctrl+C or crash loses at most the line being written at that exact moment.

---

## Known Issues and Fixes Applied

**Discord DAVE end-to-end encryption**
Discord mandated E2EE for all voice channels in early 2026, breaking every Python voice receive library. This bot uses a specific fork of `discord-ext-voice-recv` ([rdphillips7/discord-ext-voice-recv@ddd2860](https://github.com/rdphillips7/discord-ext-voice-recv)) that implements DAVE decryption, along with the `davey` package. This is pinned in `pyproject.toml`.

**Blackwell GPU (RTX 5000 series) + cuBLAS**
Blackwell requires CUDA 13.0 and the cu130 PyTorch build. Even with the correct build, `float16` compute type may fail to find `cublas64_12.dll`. Set `STT_COMPUTE_TYPE=int8_float16` in your `.env` to avoid cuBLAS entirely while keeping GPU inference.

**Whisper hallucinations**
Whisper sometimes hallucinates short phrases ("Thank you.", "Thanks for watching.") on silence or noise that passes VAD. A blocklist in `stt/transcriber.py` drops known hallucinations, and `STT_MIN_AUDIO_BYTES=8000` raises the minimum chunk size to reduce false VAD triggers.

**Ollama context window for long sessions**
Long D&D sessions (90+ minutes) can produce transcripts well over 15,000 tokens. If the transcript exceeds `num_ctx`, Ollama silently truncates the prompt — including the instructions — producing garbage output. The summarizer chunks long transcripts, summarizes each part, then combines them in a final pass. `num_ctx` is set to 16384 by default.

---

## Project Structure

```
dnd-scribe/
├── main.py              # Entry point — spawns STT process, starts bot
├── config.py            # All configuration, loaded from .env
├── .env.example         # Copy to .env and fill in your settings
├── bot/
│   ├── client.py        # Discord bot, commands, result polling, session management
│   ├── audio_sink.py    # Receives Discord audio, resamples, forwards to STT queue
│   └── summarizer.py    # Ollama integration, chunked summarization for long sessions
├── audio/
│   ├── resampler.py     # 48kHz stereo → 16kHz mono converter (torchaudio)
│   └── ring_buffer.py   # Pre-speech context buffer for VAD
├── stt/
│   ├── processor.py     # STT subprocess main loop
│   ├── transcriber.py   # faster-whisper wrapper with hallucination filtering
│   ├── vad.py           # Silero VAD wrapper
│   └── user_state.py    # Per-user audio buffers and state
├── utils/
│   └── logging.py       # Shared logging configuration
└── sessions/            # Output directory (auto-created)
```

---

## Platform Support

This project has been developed and tested exclusively on **Windows 11**. It should work on Linux in principle — the dependencies are all cross-platform and the code doesn't use anything Windows-specific — but it has not been tested and no Linux setup instructions are provided. Things that may need attention on Linux:

- The `uv` install command differs (curl-based rather than PowerShell)
- FFmpeg is typically installed via your package manager (`apt`, `pacman`, etc.) rather than winget
- PyTorch CUDA builds and library paths may behave differently
- The multiprocessing spawn method defaults differ between Windows and Linux, which could affect the STT subprocess

Linux PRs are welcome. If you get it running, please open an issue or PR with any fixes needed.

---

## AI Assistance

This project was built with significant assistance from [Claude](https://claude.ai) (Anthropic). The initial architecture, all source code, debugging, and documentation were developed collaboratively through an iterative back-and-forth process — identifying the right libraries, working through the DAVE encryption problem, diagnosing GPU and dependency issues, and refining the summarization pipeline.

The core STT architecture was adapted from an existing open-source project (see Credits), but the D&D-specific summarization layer, session management commands, crash-safe transcript logging, and chunked multi-pass summarization for long sessions were designed with Claude and implemented by me.

---

## Credits

Core STT architecture adapted from [Leehyunbin0131/Discord-Realtime-STT-Bot](https://github.com/Leehyunbin0131/Discord-Realtime-STT-Bot).

DAVE decryption via [rdphillips7/discord-ext-voice-recv](https://github.com/rdphillips7/discord-ext-voice-recv), implementing the fix from [imayhaveborkedit/discord-ext-voice-recv#54](https://github.com/imayhaveborkedit/discord-ext-voice-recv/pull/54).