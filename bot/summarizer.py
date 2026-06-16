"""
Ollama summarization for D&D session transcripts.

Called from bot/client.py when the DM requests notes.
Runs in a thread (via asyncio.to_thread) so it never blocks the event loop.
"""

from __future__ import annotations

import requests

from config import ollama_cfg
from utils.logging import get_logger

logger = get_logger(__name__)

# Edit this prompt to add your campaign name, PC names, setting, etc.
# The more context you bake in, the better the notes will be.
SCRIBE_PROMPT = """You are a scribe for a Dungeons & Dragons campaign. \
Summarize the following session transcript into organized notes.

Include these sections (omit any with no content):

**Key Events** — Major plot points, in order
**NPCs Encountered** — Names, roles, and notable interactions
**Player Decisions** — Important choices and their outcomes
**Combat Encounters** — Fights, tactics, outcomes, deaths or near-deaths
**Loot & Rewards** — Items, gold, spell scrolls, or XP gained
**Locations Visited** — Where the party went
**Open Threads** — Unresolved questions, dangling hooks, cliffhangers
**DM Notes** — Anything that might need follow-up next session

Be concise but complete. Use bullet points within each section.
Each transcript line is formatted as [Speaker Name]: text.

Transcript:
{transcript}

Session Notes:"""


def summarize(transcript: str) -> str:
    """Send the transcript to Ollama and return structured session notes."""
    logger.info(
        "Sending %d transcript lines to Ollama (%s)...",
        transcript.count("\n") + 1,
        ollama_cfg.model,
    )
    try:
        response = requests.post(
            f"{ollama_cfg.url}/api/generate",
            json={
                "model": ollama_cfg.model,
                "prompt": SCRIBE_PROMPT.format(transcript=transcript),
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 8192,
                },
            },
            timeout=ollama_cfg.timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()["response"]
        logger.info("Summarization complete.")
        return result
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama at %s", ollama_cfg.url)
        return "_Error: Could not connect to Ollama. Is it running?_"
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        return f"_Error generating summary: {exc}_"
