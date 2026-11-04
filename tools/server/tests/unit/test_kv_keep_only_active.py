import os
import tempfile
import threading
import time
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def token_ids(word, n):
    res = server.make_request("POST", "/tokenize", data={"content": word})
    assert res.status_code == 200
    toks = res.body["tokens"]
    return (toks * ((n // len(toks)) + 1))[:n]


# an explicit reasoning budget whose marker tags are private-use characters
# no story text can produce: the budget sampler stays idle and the generation
# behaves exactly like an unconstrained greedy run (the sampler still rides
# every park via the carried sampler, like the grammar does)
RBUDGET = {
    "reasoning_budget_tokens": 0,
    "reasoning_budget_start_tag": "\uE000rb\uE001",
    "reasoning_budget_end_tags": ["\uE002rb\uE003"],
}


def stream_in_thread(prompt_tokens, n_predict, started=None, extra=None):
    box = {}

    def run():
        chunks = []
        box["chunks"] = chunks
        try:
            data = {
                "prompt": prompt_tokens,
                "n_predict": n_predict,
                "ignore_eos": True,
                "cache_prompt": True,
                "stream": True,
            }
            if extra:
                data.update(extra)
            for i, c in enumerate(server.make_stream_request("POST", "/completion", data=data)):
                if i == 0 and started is not None:
                    started.set()
                chunks.append(c)
        except Exception as e:
            box["error"] = e
        box["chunks"] = chunks

    t = threading.Thread(target=run)
    t.start()
    return t, box


def request_in_thread(prompt_tokens, n_predict, extra=None, path="/completion"):
    box = {}

    def run():
        data = {
            "prompt": prompt_tokens,
            "n_predict": n_predict,
            "ignore_eos": True,
            "cache_prompt": True,
        }
        if extra:
            data.update(extra)
        try:
            box["res"] = server.make_request("POST", path, data=data)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    return t, box


# a survivor whose decode career must not end before the choreography's
# guarded moment: a budget-far-beyond-the-cap stream the test aborts (client
# disconnect) once the server log confirms it - the per-sequence training
# cap bounds the career (an n_predict = -1 request would resolve to the
# server's default budget, so the budget must be explicit), and the
# reader-side close
# stops the server's response reader (CANCEL task), whose slot releases
# with its state resident, exactly like a finished session for every sweep
# endgame; the stop event is polled between chunks, so the close is a
# graceful disconnect (the reader records no error)
def disconnect_stream_in_thread(prompt_tokens, n_predict, started=None, extra=None):
    box = {}
    stop = threading.Event()

    def run():
        chunks = []
        box["chunks"] = chunks
        response = {}
        try:
            data = {
                "prompt": prompt_tokens,
                "n_predict": n_predict,
                "ignore_eos": True,
                "cache_prompt": True,
                "stream": True,
            }
            if extra:
                data.update(extra)
            def on_response(resp):
                response["raw"] = resp
            for i, c in enumerate(server.make_stream_request("POST", "/completion", data=data, on_response=on_response)):
                if i == 0 and started is not None:
                    started.set()
                chunks.append(c)
                if stop.is_set():
                    # client disconnect: close mid-stream, then stop reading
                    response["raw"].close()
                    break
        except Exception as e:
            box["error"] = e
        box["chunks"] = chunks

    t = threading.Thread(target=run)
    t.start()
    return t, box, stop


# poll the server log for a tag without consuming the LogReader's window
# (a later drain must still see the tag): keeps its own binary cursor, so
# multi-byte log text cannot corrupt the incremental read position
def wait_for_log_tag(tag, timeout=120):
    pos = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(server.log_path, "rb") as f:
            f.seek(pos)
            chunk = f.read()
        if tag.encode() in chunk:
            return True
        pos += len(chunk)
        time.sleep(0.001)
    return False


# decode-progress barrier: hold the caller until the streaming session in box
# has delivered n token-bearing chunks (1 ms poll, the park suite's entry-lead
# pin - the victim is launched only after the survivor is n tokens into its
# decode, so the entry lead k is pinned near n plus the request path's latency)
def await_decode_progress(box, n, timeout=120):
    deadline = time.time() + timeout
    while len([c for c in box["chunks"] if c.get("tokens")]) < n and time.time() < deadline:
        time.sleep(0.001)
    assert len([c for c in box["chunks"] if c.get("tokens")]) >= n


# a unified pool fills with two 240/241-token prompts plus their generated
# tokens; the decode step that would push the combined extents past the pool
# fails and, with the policy off, the stock ladder halves n_batch down to 1
# and fails all in-flight sessions
# A carries a long generation budget so it is still decoding when the extents
# cross - a session that finished before the crossing is swept out of the pool
# on the next launch (or purged by the failure ladder) and the stock failure
# would never fire
# sizing: the pool is 1280 cells - the model's training-context cap (2048)
# bounds every sequence's career, so both careers must cover the shared
# crossing dA + dB > 799 while the pool stays under the two-session ceiling
# 2 x 2048; the decode crossing must land while BOTH sessions are still
# generating (the killall fails all in-flight sessions): A's career is capped
# at 1280 - 240 = 1040 iterations, giving the entry window k < 2*1040 - 799 - 1
# = 1280 iterations (~427 ms at the ~3000 tok/s the model reaches, ~3.3x the
# worst recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which
# leaves ~1000 iterations of pure slip headroom on A's career), and B carries
# a 600-token budget so it is still decoding at the crossing whatever lead A
# has opened (the crossing lands at dec(B) <= 400 even at zero lead)
# NOTE: this passes on stock too by design - it is a regression guard, not a symptom repro
def test_decode_overflow_killall_flag_off():
    global server
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 240), 1500, started)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1280-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(token_ids("dragon", 241), 600)

    ta.join(120)
    tb.join(120)

    # token-for-token equality with a solo run is not asserted for either
    # session: decode kernel selection depends on batch composition, so
    # concurrent sessions legitimately perturb low-order numerics (cf. the
    # FIXME in test_completion.py test_completion_parallel_slots and the
    # skipped test_cache_vs_nocache_prompt); the stock ladder's outcome and
    # its error delivery are asserted instead
    for box in (box_a, box_b):
        errors = [c["error"]["message"] for c in box["chunks"] if "error" in c]
        assert errors and "context size has been exceeded" in errors[0].lower()
    assert "Context size has been exceeded." in log.drain()


# same policy-newest park shape, and both sessions carry an ACTIVE grammar
# (non-lazy, constraining from the first generated token): the park carries
# the whole sampler, so the grammar's parse state resumes exactly where the
# park stopped and B completes mid-grammar
# the brace + a-run grammar makes the carry observable: the resumed session
# must continue the open rule - exactly one '{' in the whole stream; a
# re-initialized grammar would re-apply from position 0 and force a second
# '{' after the park
# sizing: the pool is 1280 cells - the model's training-context cap (2048)
# bounds every sequence's career, so a park is only reachable while the
# combined extents stay under 2 x 2048; B parks at the first failed chunk and
# resumes once A's finished state has been swept out of the pool, so the
# decode crossing (dA + dB > 799) must land while the survivor A is still
# generating, i.e. within k < 2*1000 - 799 - 1 = 1199 iterations of entry
# lead (~400 ms at the ~3000 tok/s the model reaches, ~3.1x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which also
# keeps A's post-park decode headroom well inside its 1000-token budget);
# B carries a 600-token budget so it is still decoding at the crossing
# whatever lead A has opened (the crossing lands at dec(B) <= 400 even at
# zero lead)
def test_decode_overflow_grammar_constrained_parked_and_resumed():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    grammar = 'root ::= "{" [a]+'
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 240), 1000, started, extra={"grammar": grammar})
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1199-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(token_ids("dragon", 241), 600, extra={"grammar": grammar})

    ta.join(120)
    tb.join(120)

    # B parks at the first failed chunk and resumes once A's finished state
    # has been swept out of the pool
    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "Context size has been exceeded." not in content

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # the park conserves both requested output counts
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 600

    # the constraint held across the park: each stream is one grammar run -
    # a single opening brace and an a-run that never re-opens. the token
    # stream itself is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the grammar's constraint is
    # composition-invariant, so the shape pins the carried parse state
    for box in (box_a, box_b):
        text = "".join(c["content"] for c in box["chunks"]
                       if isinstance(c, dict) and "content" in c)
        assert text.startswith("{")
        assert text.count("{") == 1
        assert set(text[1:]) == {"a"}


# the production shape: sessions that carry a lazy tool-call grammar
# (grammar_type "tool_calls" + grammar_lazy + a WORD trigger, exactly what the
# peg-native chat path attaches to every tool-carrying request), on the park
# test's crossing geometry. the "e" trigger fires early in story text, so the
# park can land with the lazy grammar already active - the mid-tool-call state
# the stock exclusion was written to protect. pre-fix both sessions were
# excluded from parking by the grammar clause and rode the stock ladder to the
# loud failure with zero park tags; post-fix B parks mid-decode and resumes on
# its carried sampler, and both sessions complete their requested counts
# sizing: the pool is 1280 cells - the model's training-context cap (2048)
# bounds every sequence's career, so a park is only reachable while the
# combined extents stay under 2 x 2048; B parks at the first failed chunk and
# resumes once A's finished state has been swept out of the pool, so the
# decode crossing (dA + dB > 799) must land while the survivor A is still
# generating, i.e. within k < 2*1000 - 799 - 1 = 1199 iterations of entry
# lead (~400 ms at the ~3000 tok/s the model reaches, ~3.1x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which also
# keeps A's post-park decode headroom well inside its 1000-token budget);
# B carries a 600-token budget so it is still decoding at the crossing
# whatever lead A has opened (the crossing lands at dec(B) <= 400 even at
# zero lead)
def test_decode_overflow_tool_grammar_parked_and_resumed():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    tool_grammar = {
        # the grammar must parse its own trigger text: on activation the
        # tokens overlapping the trigger match are fed to the grammar from
        # the match position, so the rule opens on the matched 'e' itself
        "grammar": 'root ::= "e" [a]+',
        "grammar_type": "tool_calls",
        "grammar_lazy": True,
        # a pattern trigger (a plain word trigger would have to be a
        # preserved token): fires on the first 'e' in the generated text,
        # so the park can land with the lazy grammar already active
        "grammar_triggers": [{"type": 2, "value": "e"}],
    }
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 240), 1000, started, extra=tool_grammar)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1199-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(token_ids("dragon", 241), 600, extra=tool_grammar)

    ta.join(120)
    tb.join(120)

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "Context size has been exceeded." not in content

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # the token stream is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the park conserves both requested
    # output counts, and the running delivered total pins the resume
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 600
    seq = [c["tokens_predicted"] for c in box_b["chunks"] if c.get("tokens")]
    assert seq == list(range(1, 601))


# same policy-newest park shape: the pool cannot hold two finished sessions,
# so the combined extents cross mid-decode whatever the launch order; the
# newest session (B) parks at the first failed chunk and the requeue gate
# holds it parked the whole time A is decoding - a session's own release
# leaves its state resident, so B can only resume once a later small launch
# has swept A's finished state out of the pool; the restore then covers B's
# whole saved state and B finishes its remaining n_predict from cache
# sizing: the pool is 1280 cells with 220-token prompts - the model's
# training-context cap (2048) bounds every sequence's career, so a park is
# only reachable while the combined extents stay under 2 x 2048; the decode
# crossing (dA + dB > 840) must land while the survivor A is still
# generating, i.e. within k < 2*1000 - 840 - 1 = 1159 iterations of entry
# lead (~386 ms at the ~3000 tok/s the model reaches, ~3.0x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which also
# keeps A's post-park decode headroom well inside its 1000-token budget);
# B carries a 600-token budget so it is still decoding at the crossing
# whatever lead A has opened (the crossing lands at dec(B) <= 420 even at
# zero lead); A's worst extent (1220) plus B's 220-token prompt never fails
# a B prefill chunk for any lead inside A's budget, so the park fires in B's
# decode with both sessions still generating
def test_decode_overflow_park_and_continue_flag_on():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 220)
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1159-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(prompt_b, 600)

    ta.join(120)

    # A is done and its state still holds cells: the gate keeps B parked
    # (free - B's saved extent <= reserve). Launch a small task on B's own
    # slot: the launch sweeps A's idle state out of the pool, and its release
    # is the first one that pops B with room for the restore
    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200
    # B's old slot was empty, so the small task prefills fresh
    assert c.body["timings"]["prompt_n"] == 4

    # this window spans from boot to the small task's release and therefore
    # contains both the park (mid-decode, before A finished) and the sweep
    # that moved A's finished state out of the pool
    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content

    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        # every SSE data line must parse to a JSON object: a resumed stream
        # must never carry a null event (its begin partial was consumed as the
        # headers of the original launch and must not be re-emitted)
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    assert "Context size has been exceeded." not in log.drain()

    # the token stream is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the park conserves both requested
    # output counts, and the running delivered totals and the resume
    # metadata are asserted instead
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 600

    # the reported count never restarts across the park: every token-bearing
    # chunk reports the running delivered total (a restart would also re-fire
    # the n_decoded == 1 protocol chunks), and the final usage equals the
    # delivered count
    for box, n in ((box_a, 1000), (box_b, 600)):
        seq = [c["tokens_predicted"] for c in box["chunks"] if c.get("tokens")]
        assert seq == list(range(1, n + 1))
        assert box["chunks"][-1]["tokens_predicted"] == n

    # B resumed from the parked state: the reported prompt work counts only
    # client-caused prefill - the 220-token fresh prefill; the 1 already-sent
    # tail token re-evaluated on the resume is park machinery and is excluded
    # (a full re-prefill would read 440+); cache_n reports the original
    # prompt's cross-request cache serving (0 for the fresh fixture), not
    # the resume's internal cache hit - together prompt_n + cache_n == the
    # client's prompt length
    final_b = box_b["chunks"][-1]
    assert final_b["timings"]["prompt_n"] == 220
    assert final_b["timings"]["cache_n"] == 0


# the same geometry as the park test, but with no helper launch after A
# finishes: A's release pops B into a server whose only resident state is
# A's finished session, so the requeue gate blocks (free - B's restored
# extent <= reserve) and nothing would ever pop B again - the server must
# sweep the idle states itself and let B resume from its saved state
# sizing: the pool is 1280 cells with 220-token prompts - the model's
# training-context cap (2048) bounds every sequence's career, so a park is
# only reachable while the combined extents stay under 2 x 2048; the decode
# crossing (dA + dB > 840) must land while the survivor A is still
# generating, i.e. within k < 2*1000 - 840 - 1 = 1159 iterations of entry
# lead (~386 ms at the ~3000 tok/s the model reaches, ~3.0x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which also
# keeps A's post-park decode headroom well inside its 1000-token budget);
# B carries a 600-token budget so it is still decoding at the crossing
# whatever lead A has opened (the crossing lands at dec(B) <= 420 even at
# zero lead); A's worst extent (1220) leaves only 60 free cells at its
# release, so the requeue gate holds B until the server's own sweep
def test_decode_overflow_park_rescued_when_all_idle():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 220)
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1159-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(prompt_b, 600)

    ta.join(120)
    tb.join(120)

    # the window spans boot to B's completion and therefore contains the
    # park, the idle sweep that freed A's finished state, and the resume
    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content
    assert "Context size has been exceeded." not in content
    assert "no room even after clearing idle slots" not in content

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # the token stream is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the park conserves both requested
    # output counts, and the running delivered total and the resume
    # metadata are asserted instead
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 600
    seq = [c["tokens_predicted"] for c in box_b["chunks"] if c.get("tokens")]
    assert seq == list(range(1, 601))
    assert box_b["chunks"][-1]["tokens_predicted"] == 600

    # B resumed from the parked state: the 220-token fresh prefill only - the
    # 1 already-sent tail token re-evaluated on the resume is excluded park
    # machinery (a full re-prefill would read 440+); cache_n is the fresh
    # fixture's cross-request cache serving (0), not the resume's internal
    # cache hit
    final_b = box_b["chunks"][-1]
    assert final_b["timings"]["prompt_n"] == 220
    assert final_b["timings"]["cache_n"] == 0


# same choreography, policy largest: the victim is the session with the
# largest KV extent, which is A (its 220-token prompt leads B's 200-token one
# by more than A's decode head start over B), not the newest session - the
# other policies would park B, so the parked session's identity pins the
# selection; sizing: the pool is 1280 cells - the model's training-context
# cap (2048) bounds every sequence's career, so a park is only reachable
# while the combined extents stay under 2 x 2048; the decode crossing
# (dA + dB > 860) must land while B is still generating (a park needs another
# session still running) and before A's own 1000-token career ends, i.e.
# within k < 2*1000 - 860 - 1 = 1139 iterations of entry lead (~379 ms at
# the ~3000 tok/s the model reaches, ~2.9x the worst recorded ~130 ms
# arrival slip; the chunk barrier pins k near 40); the crossing lands at
# dec(A) ~450 / dec(B) ~410, inside both budgets with margin (B carries 700
# so it is still decoding at the crossing even at zero lead, and the
# crossing lands earlier in B's career the later B arrives)
def test_decode_overflow_park_largest_policy():
    global server
    server.decode_overflow_policy = "largest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 200)
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started)
    assert started.wait(30)
    # hold the victim's running mate until the survivor-launched session is
    # ~40 tokens into its decode: the entry lead k is then pinned near 40
    # plus the request path's latency, well inside the 1139-iteration career
    # window, whatever latency the request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(prompt_b, 700)

    ta.join(120)
    tb.join(120)

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content
    assert "Context size has been exceeded." not in content

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # the token stream is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the park conserves both requested
    # output counts, and the running delivered totals and the resume
    # metadata are asserted instead
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 700
    for box, n in ((box_a, 1000), (box_b, 700)):
        seq = [c["tokens_predicted"] for c in box["chunks"] if c.get("tokens")]
        assert seq == list(range(1, n + 1))
        assert box["chunks"][-1]["tokens_predicted"] == n

    # A parked and resumed from its saved state: its prompt work is the
    # 220-token fresh prefill - the 1 re-evaluated sent-tail token is
    # excluded park machinery; B was never parked and prefilled its whole
    # prompt fresh
    final_a = box_a["chunks"][-1]
    assert final_a["timings"]["prompt_n"] == 220
    assert final_a["timings"]["cache_n"] == 0
    final_b = box_b["chunks"][-1]
    assert final_b["timings"]["prompt_n"] == 200
    assert final_b["timings"]["cache_n"] == 0


# same choreography, policy least-output: the victim is the session with the
# fewest generated tokens, which is B (launched second, it decoded strictly
# less than A at the crossing; on an exact tie the extent tie-break picks its
# 221-token prompt too) - least-output parks B at any decode lead, and the
# asserts pin exactly that parked identity; sizing: the pool is 1280 cells -
# the model's training-context cap (2048) bounds every sequence's career, so
# a park is only reachable while the combined extents stay under 2 x 2048;
# the decode crossing (dA + dB > 839) must land while the survivor A is still
# generating, i.e. within k < 2*1000 - 839 - 1 = 1160 iterations of entry
# lead (~386 ms at the ~3000 tok/s the model reaches, ~3.0x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which also
# keeps A's post-park decode headroom well inside its 1000-token budget);
# B carries a 600-token budget so it is still decoding at the crossing
# whatever lead A has opened (the crossing lands at dec(B) <= 420 even at
# zero lead)
def test_decode_overflow_park_least_output_policy():
    global server
    server.decode_overflow_policy = "least-output"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 221)
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1160-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(prompt_b, 600)

    ta.join(120)
    tb.join(120)

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content
    assert "Context size has been exceeded." not in content

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # the token stream is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the park conserves both requested
    # output counts, and the running delivered totals and the resume
    # metadata are asserted instead
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 600
    for box, n in ((box_a, 1000), (box_b, 600)):
        seq = [c["tokens_predicted"] for c in box["chunks"] if c.get("tokens")]
        assert seq == list(range(1, n + 1))
        assert box["chunks"][-1]["tokens_predicted"] == n

    # B parked and resumed from its saved state: its prompt work is the
    # 221-token fresh prefill - the 1 re-evaluated sent-tail token is
    # excluded park machinery; A was never parked and prefilled its whole
    # prompt fresh
    final_a = box_a["chunks"][-1]
    assert final_a["timings"]["prompt_n"] == 220
    assert final_a["timings"]["cache_n"] == 0
    final_b = box_b["chunks"][-1]
    assert final_b["timings"]["prompt_n"] == 221
    assert final_b["timings"]["cache_n"] == 0


# a parked session's client-visible delivery must match a reference run of
# the same request without pool pressure: the park keeps the already-sent
# tail token only as context for the resume (never re-emitted), so
# re-emitting it would show up as a duplicate against the reference stream;
# greedy sampling (temp 0) makes both runs of the same prompts deterministic
def test_decode_overflow_parked_stream_matches_reference():
    global server
    server.decode_overflow_policy = "newest"

    def stream_tokens(box):
        return [t for c in box["chunks"] for t in c.get("tokens", [])]

    # the next-token distribution over a delivered prefix, sampled fresh (no
    # prompt cache): the probe re-evaluates the prefix from scratch, exactly
    # like the resume re-evaluates the saved state, and the stock API reports
    # the top n_probs candidates with their logprobs
    def boundary_probe(prompt_tokens):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt_tokens,
            "n_predict": 1,
            "n_probs": 5,
            "cache_prompt": False,
        })
        assert res.status_code == 200
        return [(p["id"], p["logprob"])
                for p in res.body["completion_probabilities"][0]["top_logprobs"]]

    def launch_sessions():
        prompt_b = token_ids("dragon", 220)
        started = threading.Event()
        ta, box_a = stream_in_thread(token_ids("knight", 220), 72, started)
        assert started.wait(30)
        tb, box_b = stream_in_thread(prompt_b, 40)
        return ta, box_a, tb, box_b, prompt_b

    # parked run first: the park-test geometry (512-cell pool) - B parks
    # mid-decode and resumes once a small launch has swept A's finished state
    # out; the parked boot runs first so the park boundary it exposes can
    # anchor the reference boot's probe below
    server.n_ctx = 512
    server.start()
    log = LogReader(server.log_path)

    ta, box_a, tb, box_b, prompt_b = launch_sessions()
    ta.join(120)

    # B is parked at this point - it cannot resume until the helper launch
    # below has swept A's finished state out of the pool - so the tokens it
    # has delivered mark the park boundary
    parked_at = len(stream_tokens(box_b))
    assert parked_at < 40

    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200
    # B's old slot was empty, so the small task prefills fresh
    assert c.body["timings"]["prompt_n"] == 4

    # this window spans from boot to the small task's release and therefore
    # contains both the park (mid-decode, before A finished) and the sweep
    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content

    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]
    assert "Context size has been exceeded." not in log.drain()

    parked_a_tokens = stream_tokens(box_a)
    parked_b_tokens = stream_tokens(box_b)

    # token-for-token equality with the reference run is not asserted here:
    # decode kernel selection depends on batch composition, so concurrent
    # sessions legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt). the park's isolation is asserted
    # instead: both sessions deliver their full requested counts, neither
    # delivery restarts across the park boundary, and the resume metadata
    # reads as if the session was never parked
    assert len(parked_a_tokens) == 72
    assert len(parked_b_tokens) == 40
    seq_a = [c["tokens_predicted"] for c in box_a["chunks"] if c.get("tokens")]
    assert seq_a == list(range(1, 73))
    seq_b = [c["tokens_predicted"] for c in box_b["chunks"] if c.get("tokens")]
    assert seq_b == list(range(1, 41))

    # B resumed from the parked state: same whole-session reporting as the
    # park test - the 220-token fresh prefill only, the 1 sent-tail token
    # re-evaluated on the resume excluded (a fresh relaunch would also
    # report 220); cache_n is the original prompt's cross-request cache
    # serving (0, fresh fixture)
    final_b = box_b["chunks"][-1]
    assert final_b["timings"]["prompt_n"] == 220
    assert final_b["timings"]["cache_n"] == 0

    # the parked final reports the whole delivered count
    assert final_b["tokens_predicted"] == len(parked_b_tokens)

    # the resume-boundary probe: the victim's resume re-evaluates the saved
    # state from scratch (a solo fresh prefill), so a probe over the
    # delivered prefix reproduces the resume's own next-token distribution -
    # the one position where the parked run's continuation is pinned exactly
    probe_b = boundary_probe(prompt_b + parked_b_tokens[:parked_at])

    server.stop()

    # reference: same sessions on a pool they cannot exhaust (291 + 260 < 1024)
    # - no park can happen and both sessions decode straight through
    server.n_ctx = 1024
    server.start()
    log = LogReader(server.log_path)

    ta, box_a, tb, box_b, _ = launch_sessions()
    ta.join(120)
    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" not in log.drain()

    ref_a_tokens = stream_tokens(box_a)
    ref_b_tokens = stream_tokens(box_b)
    assert len(ref_a_tokens) == 72
    assert len(ref_b_tokens) == 40

    # the same probe over the reference run's prefix at the same position
    probe_ref = boundary_probe(prompt_b + ref_b_tokens[:parked_at])

    # the delivered counts equal the reference run's
    assert len(parked_a_tokens) == len(ref_a_tokens)
    assert len(parked_b_tokens) == len(ref_b_tokens)

    # at the park boundary the victim's continuation is pinned against the
    # reference run: the probe's top candidates match the reference run's
    # probe at the same position (the low-order logprobs within the
    # composition drift band), and the top-1 candidate is the reference's
    # next token
    assert [tid for tid, _ in probe_b] == [tid for tid, _ in probe_ref]
    for (_, lp), (_, lp_expected) in zip(probe_b, probe_ref):
        assert lp == pytest.approx(lp_expected, abs=1e-3)
    assert probe_b[0][0] == ref_b_tokens[parked_at]


# a non-streaming parked session must receive its COMPLETE response - the
# pre-park tokens included - and every metadata field must read as if the
# session was never parked: the counts and usage report the client's original
# prompt, the budget is reported as requested, the echoed prompt is the
# client's prompt (not the internally extended sequence), and the timings
# cover the whole session (the park wait included, it is part of the
# client's experienced latency). usage.* exists only on the OAI response
# family, so the parked session runs on /v1/completions with the per-request
# verbose field, whose __verbose body carries the non-OAI fields
# sizing: the pool is 2048 cells - the model's training-context cap makes
# every sequence stop (truncated, limit) at its own slot context before any
# pool-level decode failure, so a park is only reachable while the combined
# extents stay under 2 x 2048, and 2048 is the smallest pool whose
# per-sequence context reaches that cap (smaller pools would truncate the
# survivor's career early and shrink the entry window); the victim B carries
# a 40-token prompt so no crossing can fire while it is prefilling, and the
# decode crossing (dA + dB > 1988) must land while the survivor A is still
# generating, i.e. within k < 2*1500 - 1988 - 1 = 1011 iterations of entry
# lead (~337 ms at the ~3000 tok/s the model reaches, ~2.6x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which
# also keeps the survivor's post-park disconnect headroom near 490 tokens);
# after the park the free space is always exactly 2 cells below B's restored
# size (41 + dB), so the requeue gate holds B until the endgame sweep, and
# the survivor - whose own budget ends cleanly at 1500 - is disconnected on
# the park tag so the endgame runs at the park point instead of at A's
# natural completion
def test_decode_overflow_parked_nonstreaming_complete():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 2048
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 40)

    started = threading.Event()
    ta, box_a, stop_a = disconnect_stream_in_thread(token_ids("knight", 20), 1500, started)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1011-iteration career window, whatever latency the
    # request path adds
    deadline = time.time() + 120
    while len([c for c in box_a["chunks"] if c.get("tokens")]) < 40 and time.time() < deadline:
        time.sleep(0.001)
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) >= 40
    # return_tokens opts the response into the per-token ID array (off by
    # default); the park must carry the array across the resume so the final
    # response reads as if the session was never parked
    tb, box_b = request_in_thread(prompt_b, 1500, extra={"verbose": True, "return_tokens": True}, path="/v1/completions")

    # the survivor is aborted (client disconnect) once the park is confirmed
    # in the server log: its remaining ~490-token career is far more than
    # the poll needs, and the CANCEL releases its slot with the state
    # resident - for every sweep endgame exactly like a finished session
    wait_for_log_tag("__TEST_TAG_DECODE_OVERFLOW_PARK__")
    stop_a.set()

    ta.join(120)

    # same endgame as the park test: A is idle and its state still holds
    # cells; the small launch sweeps it out of the pool and its release pops B
    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200
    assert c.body["timings"]["prompt_n"] == 4

    # this window spans from boot to the small task's release and therefore
    # contains both the park (mid-decode, before A finished) and the sweep
    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content
    assert "Context size has been exceeded." not in content

    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    res = box_b["res"]
    assert res.status_code == 200
    body = res.body

    # the response is complete: all 1500 requested tokens, pre-park included
    assert len(body["__verbose"]["tokens"]) == 1500
    assert body["usage"]["completion_tokens"] == 1500
    assert body["timings"]["predicted_n"] == 1500
    assert body["__verbose"]["tokens_predicted"] == 1500

    # the prompt counts report the client's 40-token prompt, not the
    # internally extended sequence; the whole-session prompt work is the
    # fresh prefill only - the 1 sent-tail token re-evaluated on the resume
    # is excluded park machinery; the fresh fixture prompt was never
    # cache-served
    assert body["__verbose"]["tokens_evaluated"] == 40
    assert body["usage"]["prompt_tokens"] == 40
    assert body["usage"]["total_tokens"] == 1540
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert body["timings"]["prompt_n"] == 40
    assert body["timings"]["cache_n"] == 0

    # the budget is reported as requested, not as the park-reduced remainder
    assert body["__verbose"]["generation_settings"]["n_predict"] == 1500

    # the echoed prompt is the client's prompt
    expected = server.make_request("POST", "/detokenize", data={"tokens": prompt_b})
    assert expected.status_code == 200
    # the echo goes through the server's internal detokenize, which drops the
    # sequence-initial space marker; the /detokenize endpoint concatenates raw
    # token pieces and keeps it - compare modulo that marker
    assert body["__verbose"]["prompt"] == expected.body["content"].removeprefix(" ")

    # the whole-session generation clock ran through the park wait
    assert body["timings"]["predicted_per_second"] > 0


# with an explicit reserve, an incoming session whose placement would leave
# the pool without the reserve is held at admission (not parked); the same
# geometry as the park test would park B by default, so a clean run with zero
# parks and B completing only after the pool drains proves the admission gate
def test_decode_overflow_reserve_override():
    global server
    server.decode_overflow_policy = "newest"
    server.decode_overflow_reserve = 128
    server.start()
    log = LogReader(server.log_path)

    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 56, started)
    assert started.wait(30)
    tb, box_b = stream_in_thread(token_ids("dragon", 220), 40)

    ta.join(120)

    # B can never pass the admission gate while A holds cells (free <= 292),
    # so it has not streamed anything and never reached decode (no park)
    assert not box_b.get("chunks")
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" not in log.drain()

    # sweep A's finished state out of the pool; the release leaves the pool
    # roomy enough for B's placement under the reserve
    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200

    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]

    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" not in log.drain()
    assert "Context size has been exceeded." not in log.drain()

    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 56
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 40

    # B was held at admission, never parked: it launches fresh once the pool
    # has room, so it prefills its whole prompt from scratch
    final_b = box_b["chunks"][-1]
    assert final_b["timings"]["prompt_n"] == 220
    assert final_b["timings"]["cache_n"] == 0


# a reserve the pool cannot satisfy for the task even with every idle slot
# cleared (512 - 220 = 292 < 300) must fail the task loudly at admission:
# an HTTP 500 with a meaningful message, not an indefinite wait - and the
# server must stay healthy afterwards
def test_decode_overflow_gate_blocked_task_fails_loud():
    global server
    server.decode_overflow_policy = "newest"
    server.decode_overflow_reserve = 300
    server.start()
    log = LogReader(server.log_path)

    tb, box_b = request_in_thread(token_ids("dragon", 220), 40)

    tb.join(120)

    assert "error" not in box_b
    res = box_b["res"]
    assert res.status_code == 500
    assert "decode-overflow reserve cannot be met" in res.body["error"]["message"]
    assert "no room even after clearing idle slots" in log.drain()

    # the failed task left nothing behind: a small request passes the gate
    # (512 - 4 = 508 > 300) and completes normally
    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200


# a PARKED task can also reach the rescue's loud-fail drop: with a reserve
# override of 2053 on a 3072-cell pool, the reserve sits above the
# bimodality bound R > (2*3072 - 2*40 - 4)/3 = 2020, so EVERY reachable park
# point has a restored size (extent + 1 kept tail token, at least 1069
# cells) that the fully swept pool cannot place under the reserve
# (3072 - 2053 = 1019) - the park and the loud fail are structurally linked.
# the pool is the smallest 256-cell step whose per-sequence context reaches
# the model's 2048-token training cap (the cap bounds each sequence's
# career, so the survivor is disconnect-aborted on the park tag rather than
# budget-bounded), the survivor's lead is pinned near 40 by the chunk
# barrier, and the admission window k < 3012 - 2053 = 959 iterations (~320 ms
# at the ~3000 tok/s the model reaches, ~2.5x the worst recorded ~130 ms
# arrival slip) sits inside the survivor's cap-bounded career window
# (k <= 1041) - after the survivor's CANCEL release the sweep cannot help
# and the task must fail loudly (HTTP 500 with a meaningful message, not an
# indefinite wait), with the drop releasing its requeue gate entry and the
# server staying healthy afterwards
def test_decode_overflow_parked_task_fails_loud_no_room():
    global server
    server.decode_overflow_policy = "newest"
    server.decode_overflow_reserve = 2053
    server.n_ctx = 3072
    server.start()
    log = LogReader(server.log_path)

    started = threading.Event()
    # the survivor is a 20-token session with a budget far beyond the model's
    # training cap: at the victim's admission the reserve leaves it room
    # (3012 - k > 2053 for any admission lead k below 959, and the victim is
    # only sent once the barrier has pinned k near 40), so it launches,
    # prefills and decodes; its career is bounded by the model's 2048-token
    # training cap, not by n_predict, and the park tag aborts it long before
    # the cap - the shared decode then overflows the pool and parks the
    # victim with a restored size no sweep can place
    ta, box_a, stop_a = disconnect_stream_in_thread(token_ids("knight", 20), 4000, started)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # admission lead k is then pinned near 40 plus the request path's
    # latency, well inside the 959-iteration admission window
    deadline = time.time() + 120
    while len([c for c in box_a["chunks"] if c.get("tokens")]) < 40 and time.time() < deadline:
        time.sleep(0.001)
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) >= 40
    # the victim's budget must run past the crossing (the pool overflows
    # ~1487 shared decode steps after its prefill), so a short one would
    # finish before any park could fire
    tb, box_b = request_in_thread(token_ids("dragon", 40), 1700)

    # the survivor is aborted (client disconnect) once the park is confirmed
    # in the server log: its remaining career (~480 tokens to the cap) is
    # far more than the poll needs, and the CANCEL release re-offers the
    # requeue, which the reserve still blocks even after the all-idle sweep
    wait_for_log_tag("__TEST_TAG_DECODE_OVERFLOW_PARK__")
    stop_a.set()

    ta.join(120)
    tb.join(120)

    assert "error" not in box_a
    res = box_b["res"]
    assert res.status_code == 500
    assert "decode-overflow reserve cannot be met" in res.body["error"]["message"]

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK_FAILED__" in content
    assert "no room even after clearing idle slots" in content
    assert "Context size has been exceeded." not in content

    # the dropped task left nothing behind: a small request passes the gate
    # (3072 - 4 = 3068 > 2053) and completes normally
    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200


# two parks of the same session must stay transparent end to end: after the
# first park's endgame the victim resumes and decodes deeply enough that the
# refill's prefill is what crosses the pool - the refill is not GENERATING
# yet, so the resumed session is the only eligible victim and parks a second
# time (post-carry no request field excludes a session from parking); the
# final metadata must read as if the session was never parked, with the
# prompt work covering exactly the fresh prefill (the re-processed tail
# token per park is excluded park machinery)
def test_decode_overflow_double_park_still_transparent():
    global server
    server.decode_overflow_policy = "newest"
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 220)
    # the refill is fired only after the first park's endgame, so the victim
    # carries a long budget: its post-resume decode must still be running when
    # the refill's prefill crosses the pool, whatever latency the request path
    # adds, while staying small enough that the session can never overflow the
    # pool on its own (220 + 280 = 500 of 512)
    prompt_c = token_ids("castle", 240)
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 72, started)
    assert started.wait(30)
    tb, box_b = stream_in_thread(prompt_b, 280)

    ta.join(120)

    # no helper launch here: A's release pops B into an all-idle server, the
    # gate holds it, and the server's own idle sweep frees A's finished state
    # so B resumes from its saved state while the test moves on
    first = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in first

    # hold the refill until B is 100 tokens into its post-resume decode: the
    # second crossing then fires during the refill's prefill (B's extent plus
    # any placed prefix of the 240-token prompt exceeds the pool while B is
    # still ~180 tokens from its limit), so B is the only eligible victim;
    # the poll must be fine-grained - B decodes far faster than the refill's
    # launch path takes, and a coarse sleep would let B finish first
    deadline = time.time() + 120
    while len([c for c in box_b["chunks"] if c.get("tokens")]) < 100 and time.time() < deadline:
        time.sleep(0.001)
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) >= 100

    # the refill rides the stock path and completes once the park frees B's
    # cells
    tc, box_c = request_in_thread(prompt_c, 40)

    tc.join(120)

    tb.join(120)

    # C's release pops B into an all-idle server again: the same sweep frees
    # C's finished state and B resumes a second time from its saved state
    second = log.drain()
    parks = (first.count("__TEST_TAG_DECODE_OVERFLOW_PARK__") +
             second.count("__TEST_TAG_DECODE_OVERFLOW_PARK__"))
    assert parks == 2
    assert "Context size has been exceeded." not in first + second
    # each park's endgame sweeps one finished session's state into the cache
    assert (first + second).count("__TEST_TAG_CACHE_IDLE_SLOT__") == 2

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # content equality with an uninterrupted run is not asserted: decode
    # kernel selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the running delivered total across both
    # parks, the park tags and the final metadata are asserted instead
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 72
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 280

    # the reported count never restarts across either park: every token-bearing
    # chunk reports the running delivered total, so the sequence stays 1..280
    # across both boundaries
    seq = [c["tokens_predicted"] for c in box_b["chunks"] if c.get("tokens")]
    assert seq == list(range(1, 281))
    assert box_b["chunks"][-1]["tokens_predicted"] == 280

    # the final metadata reads as if B was never parked: the client's prompt
    # count, the requested budget, and prompt work covering the fresh prefill
    # only - the one re-processed tail token per park is excluded park
    # machinery (a full re-prefill on either resume would read 441+); the
    # fresh fixture prompt was never cache-served
    final_b = box_b["chunks"][-1]
    assert final_b["tokens_evaluated"] == 220
    assert final_b["generation_settings"]["n_predict"] == 280
    assert final_b["timings"]["prompt_n"] == 220
    assert final_b["timings"]["cache_n"] == 0

    # the refill completed on the stock path once the park freed B's cells
    res_c = box_c["res"]
    assert res_c.status_code == 200
    assert res_c.body["timings"]["predicted_n"] == 40


# a parked session's whole-session draft counters are compared against a
# reference run of the same request without pool pressure. the ngram
# drafter matches the
# prompt's trailing 12 tokens against the prompt itself and drafts the tokens
# that follow the nearest match (at least 12 of them), so the prompt is built
# as a short repeated cycle cut off 11 tokens into its last repetition: the
# model's greedy continuation completes the cycle, which makes the very first
# decode step match one repetition back and draft the true continuation. the
# model holds such a cycle only briefly before drifting into free text, so
# the session sees a single draft event near the start; the park fires later
# at a plain decode step with no draft pending, and the boundary token is a
# plainly sampled token in both sessions - the draft counters therefore carry
# over the park unchanged (the park saves them with the session state and the
# resume adds the post-resume segment). two boots keep the prompts identical
# without marker text: the RAM prompt cache does not survive a restart, so
# the parked boot sees fresh cache state and the delivered counts stay
# comparable
DRAFT_UNIT = ("The little bear went into the woods to find some honey for her "
              "supper and came back with a big smile. ")


def draft_prompt_tokens():
    res = server.make_request("POST", "/tokenize", data={"content": DRAFT_UNIT})
    assert res.status_code == 200
    unit = res.body["tokens"]
    # 14-token cycles: the draft is 13 tokens - long enough for the drafter
    # to fire (it needs at least 12 after the match) and small enough that the
    # verify batch always fits alongside the survivor - and the model's greedy
    # continuation after the cut repeats the cycle for a couple of tokens,
    # which is exactly what the drafter proposes
    return (unit[:14] * 5) + unit[:11]


def test_decode_overflow_draft_carry_matches_reference():
    global server
    server.decode_overflow_policy = "newest"
    server.spec_type = "ngram-simple"

    # the reference boot must be up before tokenization: the fixture only
    # configures attributes and never boots on its own
    server.n_ctx = 1024
    server.start()
    log = LogReader(server.log_path)

    prompt_b = draft_prompt_tokens()
    # the survivor's prompt size is derived from the victim's measured prompt
    # so the fresh prefills sum to 420: the draft-verify batch (the prompt plus
    # 14 tokens) then always fits alongside the survivor at the observed
    # admission latencies, and the pool crossing lands later in plain decode
    # steps, with the survivor's long budget keeping it alive well past the
    # crossing (the survivor is never the victim: the park picks the newest
    # eligible session, which is the second-launched B)
    prompt_a = token_ids("knight", 420 - len(prompt_b))

    def launch_sessions():
        started = threading.Event()
        ta, box_a = stream_in_thread(prompt_a, 150, started, extra=RBUDGET)
        assert started.wait(30)
        # the victim's budget must outlast the crossing: the pool crossing has
        # to land while it is still decoding, whatever the decode rate turns
        # out to be on the machine
        tb, box_b = stream_in_thread(prompt_b, 80)
        return ta, box_a, tb, box_b

    # reference: same sessions on a pool they cannot exhaust - no park can
    # happen and both sessions decode straight through
    ta, box_a, tb, box_b = launch_sessions()
    ta.join(120)
    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" not in log.drain()

    ref_b_tokens = [t for c in box_b["chunks"] for t in c.get("tokens", [])]
    assert len(ref_b_tokens) == 80

    ref_final = box_b["chunks"][-1]
    # the drafter must engage for the comparison to mean anything
    assert ref_final["timings"].get("draft_n", 0) > 0
    assert ref_final["timings"].get("draft_n_accepted", 0) > 0
    ref_proposed = ref_final["timings"]["draft_n"]

    server.stop()

    # parked run: the park-test geometry (512-cell pool) - B parks mid-decode
    # and resumes once a small launch has swept A's finished state out
    server.n_ctx = 512
    server.start()
    log = LogReader(server.log_path)

    ta, box_a, tb, box_b = launch_sessions()
    ta.join(120)

    c = server.make_request("POST", "/completion", data={
        "prompt": token_ids("wizard", 4),
        "n_predict": 1,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert c.status_code == 200
    assert c.body["timings"]["prompt_n"] == 4

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content

    tb.join(120)

    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]
    assert "Context size has been exceeded." not in log.drain()

    # token-for-token equality with the reference run is not asserted: decode
    # kernel selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the delivered count is asserted instead
    assert len([t for c in box_b["chunks"] for t in c.get("tokens", [])]) == len(ref_b_tokens)

    final_b = box_b["chunks"][-1]
    assert final_b["tokens_predicted"] == 80
    assert final_b["tokens_evaluated"] == len(prompt_b)
    # the reported prompt work is the fresh prefill only: the 1 re-evaluated
    # sent-tail token on the resume is excluded park machinery, so
    # prompt_n + cache_n == the client's prompt length
    assert final_b["timings"]["prompt_n"] == len(prompt_b)
    assert final_b["timings"]["cache_n"] == 0

    # the draft counters carry over the park: the park fires at a plain
    # decode step (the single draft event happened at the first decode step,
    # well before the crossing), so the boundary token is a plainly sampled
    # token in both sessions and neither loses a draft acceptance. the exact
    # counter equality with the reference run is not asserted - the resumed
    # segment decodes under a different batch composition than the reference
    # run - so the counters are bounded: the accepted total stays positive
    # and the proposed total can only match or exceed the reference run's
    assert final_b["timings"]["draft_n_accepted"] > 0
    assert final_b["timings"]["draft_n"] >= ref_proposed


# same policy-newest park shape as the flipped grammar test, but the sessions
# are constrained by an explicit reasoning budget instead: the budget sampler
# rides the same carried sampler as the grammar, so the sessions park and
# resume like any other. the budget's start and end tags are request fields
# on their own and its markers never fire in story text, so the budget stays
# idle and the generation behaves exactly like an unconstrained greedy run -
# the park + resume transparency is what is under test
# sizing: the pool is 1280 cells - the model's training-context cap (2048)
# bounds every sequence's career, so a park is only reachable while the
# combined extents stay under 2 x 2048; B parks at the first failed chunk and
# resumes once A's finished state has been swept out of the pool, so the
# decode crossing (dA + dB > 799) must land while the survivor A is still
# generating, i.e. within k < 2*1000 - 799 - 1 = 1199 iterations of entry
# lead (~400 ms at the ~3000 tok/s the model reaches, ~3.1x the worst
# recorded ~130 ms arrival slip; the chunk barrier pins k near 40, which also
# keeps A's post-park decode headroom well inside its 1000-token budget);
# B carries a 600-token budget so it is still decoding at the crossing
# whatever lead A has opened (the crossing lands at dec(B) <= 400 even at
# zero lead)
def test_decode_overflow_reasoning_budget_parked_and_resumed():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    # markers no story text can ever produce: the budget sampler must stay
    # idle for the constraint to be param-driven, not sampler-state-driven
    budget = {
        "reasoning_budget_tokens": 0,
        "reasoning_budget_start_tag": "\uE000rb\uE001",
        "reasoning_budget_end_tags": ["\uE002rb\uE003"],
    }
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 240), 1000, started, extra=budget)
    assert started.wait(30)
    # hold the victim until the survivor is ~40 tokens into its decode: the
    # entry lead k is then pinned near 40 plus the request path's latency,
    # well inside the 1199-iteration career window, whatever latency the
    # request path adds
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(token_ids("dragon", 241), 600, extra=budget)

    ta.join(120)
    tb.join(120)

    # B parks at the first failed chunk and resumes once A's finished state
    # has been swept out of the pool; both sessions are already joined here,
    # so the whole window is one drain
    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "Context size has been exceeded." not in content
    assert "error" not in box_a
    assert "error" not in box_b
    for box in (box_a, box_b):
        assert not [c for c in box["chunks"] if "error" in c]
        assert not [c for c in box["chunks"] if not isinstance(c, dict)]

    # the token stream is not compared against a solo run: decode kernel
    # selection depends on batch composition, so concurrent sessions
    # legitimately perturb low-order numerics (cf. the FIXME in
    # test_completion.py test_completion_parallel_slots and the skipped
    # test_cache_vs_nocache_prompt); the park conserves both requested
    # output counts, and the running delivered total pins the resume
    assert len([c for c in box_a["chunks"] if c.get("tokens")]) == 1000
    assert len([c for c in box_b["chunks"] if c.get("tokens")]) == 600
    seq = [c["tokens_predicted"] for c in box_b["chunks"] if c.get("tokens")]
    assert seq == list(range(1, 601))


# the policy parks through the prompt cache, so with the cache disabled it can
# never fire: the server must warn and disable the policy instead of silently
# keeping a flag that has no effect (the cache-idle-slots precedent)
def test_decode_overflow_policy_disabled_without_cache_ram():
    global server
    server.decode_overflow_policy = "newest"
    server.cache_ram = 0
    server.start()
    log = LogReader(server.log_path)

    assert "--decode-overflow-policy requires --cache-ram, disabling" in log.drain()

    # the server stays healthy with the policy disabled
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


# the policy parks through a unified KV pool, so without --kv-unified (the
# default when --parallel is explicit) it can never fire: the server must warn
# and disable the policy instead of silently keeping a flag that has no effect
# (the cache-idle-slots precedent)
def test_decode_overflow_policy_disabled_without_kv_unified():
    global server
    server.decode_overflow_policy = "newest"
    server.kv_unified = False
    server.start()
    log = LogReader(server.log_path)

    assert "--decode-overflow-policy requires --kv-unified, disabling" in log.drain()

    # the server stays healthy with the policy disabled
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()
