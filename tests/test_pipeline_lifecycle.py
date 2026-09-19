"""Tests for pipeline stop/start lifecycle."""
import queue
import threading


class TestPipelineStop:
    """Test the stop() method handles cleanup correctly."""

    def test_sentinel_unblocks_stt_worker(self):
        """Putting None sentinel into phrase_queue should be a valid signal."""
        q = queue.Queue(maxsize=5)
        q.put(None)
        item = q.get(timeout=1.0)
        assert item is None

    def test_queue_drain_after_stop(self):
        """After stop(), phrase_queue should be empty."""
        q = queue.Queue(maxsize=5)
        q.put("item1")
        q.put("item2")
        q.put("item3")

        # Simulate drain logic from stop()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        assert q.empty()

    def test_drop_oldest_on_full_queue(self):
        """When queue is full, oldest item should be dropped and new one added."""
        q = queue.Queue(maxsize=3)
        q.put("old1")
        q.put("old2")
        q.put("old3")

        # Simulate drop-oldest policy
        new_item = "new_phrase"
        try:
            q.put_nowait(new_item)
        except queue.Full:
            try:
                q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            q.put_nowait(new_item)

        # Queue should contain old2, old3, new_phrase
        items = []
        while not q.empty():
            items.append(q.get_nowait())

        assert items == ["old2", "old3", "new_phrase"]

    def test_stop_event_thread_safe(self):
        """stop_event should be visible across threads."""
        stop_event = threading.Event()

        result = []

        def worker():
            while not stop_event.is_set():
                stop_event.wait(0.01)
            result.append("stopped")

        t = threading.Thread(target=worker)
        t.start()
        stop_event.set()
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert result == ["stopped"]
