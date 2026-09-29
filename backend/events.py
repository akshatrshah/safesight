"""In-process event bus that fans pipeline events out to live subscribers (SSE clients).

The worker publishes from a plain thread, subscribers consume from the asyncio event
loop, so publish() hops onto the subscriber's loop with call_soon_threadsafe.
A slow subscriber's queue is bounded, and on overflow the oldest event is dropped
for that subscriber only, one stalled browser tab must never back-pressure the pipeline.
"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict

QUEUE_SIZE = 500


class Subscription:
    def __init__(self, topic: str, loop: asyncio.AbstractEventLoop) -> None:
        self.topic = topic
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    def _put(self, event: dict) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)


class EventBus:
    """Topics are job ids, plus the special topic "*" that receives every event."""

    ALL = "*"

    def __init__(self) -> None:
        self._subs: dict[str, set[Subscription]] = defaultdict(set)
        self._lock = threading.Lock()

    def subscribe(self, topic: str) -> Subscription:
        sub = Subscription(topic, asyncio.get_running_loop())
        with self._lock:
            self._subs[topic].add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs[sub.topic].discard(sub)
            if not self._subs[sub.topic]:
                del self._subs[sub.topic]

    def publish(self, job_id: str, event: dict) -> None:
        with self._lock:
            targets = list(self._subs.get(job_id, ())) + list(self._subs.get(self.ALL, ()))
        for sub in targets:
            try:
                sub.loop.call_soon_threadsafe(sub._put, event)
            except RuntimeError:  # subscriber's loop already closed
                self.unsubscribe(sub)
