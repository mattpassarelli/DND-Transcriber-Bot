from __future__ import annotations

import queue
import time
from multiprocessing import Queue as MPQueue
from typing import Optional, Union

import discord
from discord.ext.voice_recv import AudioSink, VoiceData

from audio.resampler import AudioResampler
from utils.logging import get_logger

logger = get_logger(__name__)


class STTAudioSink(AudioSink):

    def __init__(self, audio_queue: MPQueue) -> None:
        super().__init__()
        self._queue = audio_queue
        self._resampler = AudioResampler()
        self._dropped_frames = 0
        self._last_drop_log = 0.0
        self._packet_count = 0
        self._write_count = 0
        logger.info("STTAudioSink initialised.")

    def wants_opus(self) -> bool:
        return False

    def write(
        self,
        user: Optional[Union[discord.User, discord.Member]],
        data: VoiceData,
    ) -> None:
        self._write_count += 1
        if self._write_count <= 5:
            logger.info("write() called — user=%s, pcm_len=%s", user, len(data.pcm) if data.pcm else None)

        if user is None:
            logger.warning("write() called with user=None")
            return
        try:
            resampled = self._resampler.resample(data.pcm)
            self._queue.put_nowait((user.id, resampled))
        except queue.Full:
            self._dropped_frames += 1
            now = time.monotonic()
            if now - self._last_drop_log >= 5:
                logger.warning("Audio queue full; dropped %d frames.", self._dropped_frames)
                self._dropped_frames = 0
                self._last_drop_log = now
        except Exception as exc:
            logger.error("AudioSink write error: %s", exc)

    def cleanup(self) -> None:
        logger.info("STTAudioSink cleanup. write() was called %d times total.", self._write_count)