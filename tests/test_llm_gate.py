"""Fair admission queue in front of the chat LLM."""

from __future__ import annotations

import threading
import time

import pytest

from lab_chatbot.errors import BackendBusy, ChatbotError
from lab_chatbot.llm import OllamaLLM
from lab_chatbot.llm_gate import LLMGate, get_gate, listening


def wait_until(predicate, timeout=5.0):  # generous: CI machines can be slow
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.002)
    return False


def test_fifo_order_with_limit_one():
    gate = LLMGate(1)
    order, release = [], threading.Event()

    def worker(name):
        with gate.slot(5):
            order.append(name)
            if name == 0:
                release.wait(2)

    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
        assert wait_until(lambda: gate.waiting + gate.active == i + 1)
    release.set()
    for t in threads:
        t.join(2)
    assert order == [0, 1, 2, 3]
    assert gate.active == 0 and gate.waiting == 0


def test_limit_two_runs_two_at_once():
    gate = LLMGate(2)
    release, running, peak = threading.Event(), [0], [0]
    lock = threading.Lock()

    def worker():
        with gate.slot(5):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            release.wait(2)
            with lock:
                running[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    assert wait_until(lambda: gate.active == 2 and gate.waiting == 2)
    release.set()
    for t in threads:
        t.join(2)
    assert peak[0] == 2


def test_position_callbacks():
    gate = LLMGate(1)
    seen = []
    release = threading.Event()

    def holder():
        with gate.slot(5):
            release.wait(2)

    h = threading.Thread(target=holder)
    h.start()
    assert wait_until(lambda: gate.active == 1)

    def waiter(tag):
        with listening(lambda pos, active: seen.append((tag, pos, active))):
            with gate.slot(5):
                if tag == "a":
                    # b must see itself move up to position 1 while a runs
                    wait_until(lambda: ("b", 1, 1) in seen)

    a = threading.Thread(target=waiter, args=("a",))
    a.start()
    assert wait_until(lambda: gate.waiting == 1)
    b = threading.Thread(target=waiter, args=("b",))
    b.start()
    assert wait_until(lambda: gate.waiting == 2)
    release.set()
    for t in (h, a, b):
        t.join(2)
    assert ("a", 1, 1) in seen and ("b", 2, 1) in seen
    assert ("b", 1, 1) in seen  # moved up after a was admitted
    assert [s for s in seen if s[0] == "a"][-1][1] == 0  # cleared on admission
    assert [s for s in seen if s[0] == "b"][-1][1] == 0


def test_timeout_raises_backend_busy_and_frees_ticket():
    gate = LLMGate(1)
    with gate.slot(1):
        with pytest.raises(BackendBusy):
            with gate.slot(0.05):
                pass
        assert gate.waiting == 0
    assert issubclass(BackendBusy, ChatbotError)
    with gate.slot(0.1):  # still usable
        pass


def test_get_gate_is_cached():
    assert get_gate("http://x", 1) is get_gate("http://x", 1)
    assert get_gate("http://x", 1) is not get_gate("http://x", 2)


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail

    def chat(self, **kwargs):
        if kwargs.get("stream"):
            def gen():
                yield {"message": {"content": "a"}}
                if self.fail:
                    raise ValueError("boom")
                yield {"message": {"content": "b"}}
                yield {"message": {"content": "c"}}

            return gen()
        return {"message": {"content": "done"}}


def make_llm(gate, fail=False):
    llm = OllamaLLM(gate=gate, queue_timeout=1)
    llm._client = FakeClient(fail)
    return llm


def test_complete_holds_and_releases():
    gate = LLMGate(1)
    llm = make_llm(gate)
    seen = []
    orig = llm._client.chat
    llm._client.chat = lambda **kw: (seen.append(gate.active), orig(**kw))[1]
    assert llm.complete([{"role": "user", "content": "x"}]) == "done"
    assert seen == [1] and gate.active == 0


def test_stream_holds_slot_lazily_and_releases_on_exhaustion():
    gate = LLMGate(1)
    stream = make_llm(gate).stream([])
    assert gate.active == 0  # not acquired before first iteration
    assert next(stream) == "a"
    assert gate.active == 1
    assert list(stream) == ["b", "c"]
    assert gate.active == 0


def test_stream_releases_on_early_close():
    gate = LLMGate(1)
    stream = make_llm(gate).stream([])
    next(stream)
    assert gate.active == 1
    stream.close()
    assert gate.active == 0


def test_stream_releases_on_exception():
    gate = LLMGate(1)
    stream = make_llm(gate, fail=True).stream([])
    assert next(stream) == "a"
    with pytest.raises(ValueError):
        next(stream)
    assert gate.active == 0
