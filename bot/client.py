"""
Discord bot client — commands, result handling, and D&D session management.

Built on the same architecture as Leehyunbin0131/Discord-Realtime-STT-Bot,
with D&D-specific summarization commands layered on top.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from datetime import datetime
from multiprocessing import Queue as MPQueue
from typing import cast

import discord
import discord.ext.voice_recv
from discord.ext import commands

from bot.audio_sink import STTAudioSink
from bot.summarizer import summarize
from config import discord_cfg, process_cfg
from utils.logging import get_logger

logger = get_logger(__name__)


class STTBot:
    """Wraps a commands.Bot with STT + D&D session lifecycle management."""

    def __init__(
        self,
        audio_queue: MPQueue,
        result_queue: MPQueue,
        command_queue: MPQueue,
    ) -> None:
        self._audio_q = audio_queue
        self._result_q = result_queue
        self._cmd_q = command_queue
        self._result_task: asyncio.Task | None = None

        # Session state
        self._transcript: list[str] = []
        self._user_names: dict[int, str] = {}  # user_id → display name cache
        self._session_start: datetime | None = None

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # Needed to resolve user_id → display name

        self._bot = commands.Bot(
            command_prefix=discord_cfg.command_prefix,
            intents=intents,
        )
        self._register_events()
        self._register_commands()

    def run(self) -> None:
        if not discord_cfg.token:
            logger.critical("DISCORD_TOKEN is not set. Aborting.")
            return
        self._bot.run(discord_cfg.token, log_handler=None)

    # ── Events ─────────────────────────────────────────────────────────────────

    def _register_events(self) -> None:
        bot = self._bot

        @bot.event
        async def on_ready() -> None:
            logger.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
            logger.info(
                "Commands: %sjoin  %sleave  %ssummary  %sclearnotes  %sstatus",
                *([discord_cfg.command_prefix] * 5),
            )
            if self._result_task is None or self._result_task.done():
                self._result_task = bot.loop.create_task(self._poll_results())

        @bot.event
        async def on_voice_state_update(
            member: discord.Member,
            before: discord.VoiceState,
            after: discord.VoiceState,
        ) -> None:
            if before.channel == after.channel:
                return
            if bot.user and member.id == bot.user.id and before.channel:
                self._send_command("FLUSH_ALL", None)
                return
            if before.channel:
                self._send_command("LEAVE", member.id)

    # ── Commands ────────────────────────────────────────────────────────────────

    def _register_commands(self) -> None:
        bot = self._bot

        @bot.command(name="join")
        async def cmd_join(ctx: commands.Context) -> None:
            """Join the author's voice channel and start transcribing."""
            member = cast(discord.Member, ctx.author)
            if not member.voice or not member.voice.channel:
                await ctx.send("❌ You are not in a voice channel.")
                return

            channel = member.voice.channel

            # Cache display names of everyone already in the channel
            for m in channel.members:
                self._user_names[m.id] = m.display_name

            if ctx.voice_client is not None:
                await ctx.voice_client.move_to(channel)
            else:
                vc = await channel.connect(
                    cls=discord.ext.voice_recv.VoiceRecvClient,
                )
                vc.listen(STTAudioSink(self._audio_q))
                async def _debug():
                    await asyncio.sleep(6)
                    conn = vc._connection
                    logger.info("SSRC map: %s", getattr(conn, 'ssrc_map', 'NOT FOUND'))
                    logger.info("Voice client type: %s", type(vc))
                    logger.info("Connection type: %s", type(conn))
                    logger.info("Connection attrs: %s", [a for a in dir(conn) if 'ssrc' in a.lower() or 'user' in a.lower()])
                asyncio.ensure_future(_debug())

            self._session_start = datetime.now()
            await ctx.send(
                f"🎲 Joined **{channel.name}** — session started at "
                f"{self._session_start.strftime('%H:%M')}.\n"
                f"I'm transcribing everything. Use `{discord_cfg.command_prefix}summary` for notes."
            )
            logger.info("Joined voice channel: %s", channel)


        @bot.command(name="leave")
        async def cmd_leave(ctx: commands.Context) -> None:
            """Flush audio, post final notes, and leave."""
            if not ctx.voice_client:
                await ctx.send("❌ I am not in a voice channel.")
                return

            await ctx.send("📜 Wrapping up — generating final session notes...")
            self._send_command("FLUSH_ALL", None)
            await ctx.voice_client.disconnect()

            text_channel = cast(discord.TextChannel, ctx.channel)
            await _post_summary(self, text_channel, final=True)

            if self._session_start:
                elapsed = datetime.now() - self._session_start
                minutes = int(elapsed.total_seconds() // 60)
                await ctx.send(f"👋 Left the voice channel. Session ran for {minutes} minutes.")
                self._session_start = None

            logger.info("Left voice channel.")

        @bot.command(name="summary")
        async def cmd_summary(ctx: commands.Context) -> None:
            """Post session notes without stopping the recording."""
            await ctx.send("📜 Generating session notes...")
            text_channel = cast(discord.TextChannel, ctx.channel)
            await _post_summary(self, text_channel, final=False)

        @bot.command(name="clearnotes")
        async def cmd_clearnotes(ctx: commands.Context) -> None:
            """Wipe the transcript and start fresh."""
            self._transcript.clear()
            await ctx.send("🗑️ Transcript cleared. Starting fresh from this point.")

        @bot.command(name="status")
        async def cmd_status(ctx: commands.Context) -> None:
            """Show current recording status."""
            lines = len(self._transcript)
            if self._session_start:
                elapsed = datetime.now() - self._session_start
                minutes = int(elapsed.total_seconds() // 60)
                await ctx.send(
                    f"📊 Recording for {minutes} min — {lines} transcript lines captured."
                )
            else:
                await ctx.send(
                    f"📊 Not currently recording. {lines} transcript lines in memory."
                )

    def _send_command(self, command: str, payload: object) -> None:
        try:
            self._cmd_q.put_nowait((command, payload))
        except queue.Full:
            logger.warning("Command queue is full; dropped %s command.", command)
        except Exception as exc:
            logger.error("Failed to send %s command: %s", command, exc)

    # ── Result polling ──────────────────────────────────────────────────────────

    async def _poll_results(self) -> None:
        """Drain the result queue, log transcriptions, and append to session transcript."""
        logger.info("Result polling task started.")
        while True:
            try:
                drained = False
                while True:
                    try:
                        result = self._result_q.get_nowait()
                    except queue.Empty:
                        break
                    drained = True

                    user_id = result["user_id"]
                    text = result["text"]
                    speaker = self._user_names.get(user_id, str(user_id))

                    line = f"[{speaker}]: {text}"
                    self._transcript.append(line)

                    logger.info("📝 %s  (latency %s)", line, result["latency"])
                    print(json.dumps({**result, "speaker": speaker}, ensure_ascii=False))

                if drained:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(process_cfg.result_poll_interval)
            except Exception as exc:
                logger.error("Result poll error: %s", exc)
                await asyncio.sleep(1)


# ── Summary helper ──────────────────────────────────────────────────────────────

async def _post_summary(bot: STTBot, channel: discord.TextChannel, final: bool) -> None:
    transcript_text = "\n".join(bot._transcript)

    if not transcript_text.strip():
        await channel.send("⚠️ No transcript yet — nothing to summarize.")
        return

    summary = await asyncio.to_thread(summarize, transcript_text)

    os.makedirs("sessions", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = "final" if final else "checkpoint"
    filename = f"sessions/session_{timestamp}_{label}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# D&D Session Notes\n")
        f.write(f"_{datetime.now().strftime('%B %d, %Y at %H:%M')}_\n\n")
        f.write(summary)

    header = "## 📜 Final Session Notes" if final else "## 📜 Session Notes (Checkpoint)"
    await channel.send(header)

    chunks = [summary[i : i + 1900] for i in range(0, len(summary), 1900)]
    for chunk in chunks:
        await channel.send(chunk)

    await channel.send(f"_(Saved to `{filename}`)_")

