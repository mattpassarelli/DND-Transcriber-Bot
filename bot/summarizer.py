"""
Ollama summarization for D&D session transcripts.

Called from bot/client.py when the DM requests notes.
Runs in a thread (via asyncio.to_thread) so it never blocks the event loop.

Long sessions are chunked and summarized in passes, then combined with a
final pass, to avoid exceeding the model's context window. A couple hours
of D&D talk easily produces 15-30k+ tokens of transcript, well beyond a
small context window — silent truncation there is what produces garbage
one-word replies like "this".
"""

from __future__ import annotations

import requests

from config import ollama_cfg
from utils.logging import get_logger

logger = get_logger(__name__)

# Edit this to add your campaign name, PC names, setting, etc.
# The more context you bake in, the better the notes will be.
CAMPAIGN_CONTEXT = ""  # e.g. "The party: Aldric (paladin), Zessa (warlock)..."

SINGLE_PASS_PROMPT = """/no_think
You are a scribe for a Dungeons & Dragons campaign. \
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

{campaign_context}
Transcript:
{transcript}

Session Notes:"""

CHUNK_PROMPT = """/no_think
You are a scribe for a Dungeons & Dragons campaign. \
The following is ONE PART of a longer session transcript (not the whole session).
Extract and list anything notable from THIS PART ONLY. Be concise — short bullet points.

Include only categories that have content:
- Key Events
- NPCs Encountered
- Player Decisions
- Combat Encounters
- Loot & Rewards
- Locations Visited
- Open Threads

{campaign_context}
Each transcript line is formatted as [Speaker Name]: text.

Transcript part:
{transcript}

Notes for this part:"""

FINAL_PROMPT = """/no_think
You are a scribe for a Dungeons & Dragons campaign. \
Below are notes extracted from successive parts of a single session, in order.
Combine them into ONE clean, de-duplicated set of session notes.

Structure the output with these sections (omit any with no content):

**Key Events** — Major plot points, in chronological order
**NPCs Encountered** — Names, roles, and notable interactions
**Player Decisions** — Important choices and their outcomes
**Combat Encounters** — Fights, tactics, outcomes, deaths or near-deaths
**Loot & Rewards** — Items, gold, spell scrolls, or XP gained
**Locations Visited** — Where the party went
**Open Threads** — Unresolved questions, dangling hooks, cliffhangers
**DM Notes** — Anything that might need follow-up next session

Merge duplicate mentions, keep it concise, use bullet points within each section.

{campaign_context}
Notes from each part:
{partial_notes}

Final combined session notes:"""

# Conservative chunk size in characters (~4 chars/token). Keeps each chunk
# comfortably under NUM_CTX after prompt overhead, with room for the
# model's own response.
CHUNK_CHAR_LIMIT = 12_000

# Context window passed to Ollama for every call. The previous default of
# 8192 was too small — a long session's transcript alone could exceed it,
# causing Ollama to silently truncate the prompt (including instructions),
# which is why summaries came back as garbage like "this". Chunking plus
# this larger window fixes it. Increase further if your GPU has VRAM to
# spare; num_ctx scales KV-cache memory roughly linearly.
NUM_CTX = 16384


def _call_ollama(prompt: str, num_ctx: int = NUM_CTX) -> str:
    response = requests.post(
        f"{ollama_cfg.url}/api/generate",
        json={
            "model": ollama_cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": num_ctx,
            },
        },
        timeout=ollama_cfg.timeout_seconds,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _chunk_transcript(transcript: str, limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """Split transcript into chunks on line boundaries, never mid-line."""
    lines = transcript.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def summarize(transcript: str) -> str:
    """
    Send the transcript to Ollama and return structured session notes.

    Transcripts longer than CHUNK_CHAR_LIMIT are split into chunks,
    summarized individually, then combined in a final pass. This avoids
    silently exceeding num_ctx, which causes the model to lose the
    instructions and produce garbage output.
    """
    context_block = f"Campaign context: {CAMPAIGN_CONTEXT}\n" if CAMPAIGN_CONTEXT else ""

    chunks = _chunk_transcript(transcript)
    logger.info(
        "Summarizing %d transcript lines across %d chunk(s) via %s (num_ctx=%d)...",
        transcript.count("\n") + 1,
        len(chunks),
        ollama_cfg.model,
        NUM_CTX,
    )

    try:
        if len(chunks) == 1:
            # Short session — single pass with the full structured prompt.
            prompt = SINGLE_PASS_PROMPT.format(
                campaign_context=context_block,
                transcript=chunks[0],
            )
            result = _call_ollama(prompt)
            logger.info("Summarization complete (single pass).")
            return result

        # Multi-chunk: extract notes from each part first.
        partial_notes = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Summarizing chunk %d/%d...", i, len(chunks))
            prompt = CHUNK_PROMPT.format(
                campaign_context=context_block,
                transcript=chunk,
            )
            notes = _call_ollama(prompt)
            partial_notes.append(f"--- Part {i} ---\n{notes}")

        # Combine all partial notes into one final summary.
        combined_input = "\n\n".join(partial_notes)
        logger.info("Combining %d partial summaries into final notes...", len(chunks))
        final_prompt = FINAL_PROMPT.format(
            campaign_context=context_block,
            partial_notes=combined_input,
        )
        result = _call_ollama(final_prompt)
        logger.info("Summarization complete (%d chunks combined).", len(chunks))
        return result

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama at %s", ollama_cfg.url)
        return "_Error: Could not connect to Ollama. Is it running?_"
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        return f"_Error generating summary: {exc}_"