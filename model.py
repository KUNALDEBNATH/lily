"""LILI model layer: local LLM adapter, structured-output validation, speech in, speech out.

Everything here is replaceable behind a tiny interface:
  LLM  : chat(messages, schema=None) -> str
  STT  : transcribe(float32 16 kHz mono array) -> str
  TTS  : speak(text) / is_speaking() / stop() / wait()
Nothing here needs a cloud service or API key.
"""
import difflib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms at 16 kHz


class LLMError(Exception):
    pass


class SpeechError(Exception):
    """Raised with an actionable message when audio hardware or a speech model is unavailable."""


# --------------------------------------------------------------- validation
def validate(obj, schema, path="$"):
    """Minimal JSON-schema subset validator. Returns a list of error strings (empty == valid)."""
    errs, t = [], schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "number": (int, float), "integer": int}
    if t:
        ok = isinstance(obj, types[t]) and not (t in ("number", "integer") and isinstance(obj, bool))
        if not ok:
            return [f"{path}: expected {t}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in enum")
    if t == "string":
        if len(obj) > schema.get("maxLength", 10_000):
            errs.append(f"{path}: too long")
    if t in ("number", "integer"):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: below minimum")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: above maximum")
    if t == "array":
        if len(obj) > schema.get("maxItems", 1000):
            errs.append(f"{path}: too many items")
        for i, it in enumerate(obj):
            errs += validate(it, schema.get("items", {}), f"{path}[{i}]")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}.{k}: missing")
        for k, v in obj.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: unexpected field")
    return errs


def parse_json_loose(text):
    """Extract the first JSON object from model text (handles code fences / chatter). Never uses eval."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object")


def decision_schema(tool_names):
    """The only shape of answer the agent accepts from the model."""
    return {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": list(tool_names) + ["ask_user", "finish", "give_up"]},
            "args": {"type": "object"},
            "say": {"type": "string", "maxLength": 600},
            "why": {"type": "string", "maxLength": 300},
            "expect": {"type": "object"},
        },
    }


# --------------------------------------------------------------------- LLM
class OllamaLLM:
    """Talks to a local Ollama server over plain HTTP (no extra Python dependencies)."""

    def __init__(self, model="qwen3:4b", host="http://127.0.0.1:11434", timeout=60):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def available(self):
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
        except Exception as e:  # noqa: BLE001 - report the reason, never hide it
            return False, f"Ollama not reachable at {self.host}: {e}"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return False, f"model {self.model!r} not installed (installed: {names or 'none'}). Run: ollama pull {self.model}"
        return True, "ok"

    def chat(self, messages, schema=None):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096}}
        if schema:
            body["format"] = schema
        if self.model.startswith("qwen3"):
            body["think"] = False
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as e:
            raise LLMError(f"local model call failed: {e}") from e


class ScriptedLLM:
    """Deterministic stand-in for tests and the self-test. Each reply is a dict/str/Exception or a callable."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def available(self):
        return True, "scripted"

    def chat(self, messages, schema=None):
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("scripted model has no more replies")
        r = self.replies.pop(0)
        r = r(messages) if callable(r) else r
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def decide(llm, messages, schema, retries=1):
    """Ask the model for one schema-valid decision. Returns (decision|None, error|None).
    Invalid output gets `retries` repair attempts, then a safe None; the caller must handle it."""
    msgs, err = list(messages), None
    for _ in range(retries + 1):
        try:
            raw = llm.chat(msgs, schema)
        except LLMError as e:
            return None, str(e)
        try:
            obj = parse_json_loose(raw)
            errs = validate(obj, schema)
        except (ValueError, json.JSONDecodeError) as e:
            obj, errs = None, [str(e)]
        if not errs:
            return obj, None
        err = "invalid model output: " + "; ".join(errs[:3])
        msgs = msgs + [{"role": "assistant", "content": str(raw)[:500]},
                       {"role": "user", "content": "That was invalid (" + "; ".join(errs[:3]) +
                        "). Reply again with ONLY one valid JSON object."}]
    return None, err


# --------------------------------------------------------------------- VAD
class EnergyVAD:
    """Pure-numpy voice activity detector with an adaptive noise floor.
    (Chosen over silero-vad, which would pull in torch, to keep installation light.)
    feed(frame) -> None | 'start' | 'end' | 'noise'; audio() returns the captured utterance."""

    def __init__(self, start_frames=3, end_silence_s=0.7, min_speech_s=0.25, preroll_s=0.3,
                 mult=3.0, min_rms=0.008, max_s=20.0, frame=FRAME, sr=SAMPLE_RATE):
        self.fs = frame / sr
        self.start_frames, self.end_frames = start_frames, int(end_silence_s / self.fs)
        self.min_frames, self.max_frames = int(min_speech_s / self.fs), int(max_s / self.fs)
        self.mult, self.min_rms = mult, min_rms
        self.preroll = deque(maxlen=max(1, int(preroll_s / self.fs)))
        self.floor, self.in_speech, self.loud, self.quiet, self.buf, self.voiced = 0.003, False, 0, 0, [], 0

    @staticmethod
    def rms(frame):
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0

    @property
    def threshold(self):
        return max(self.min_rms, self.floor * self.mult)

    def reset(self):
        self.in_speech, self.loud, self.quiet, self.buf, self.voiced = False, 0, 0, [], 0
        self.preroll.clear()

    def feed(self, frame):
        r = self.rms(frame)
        if not self.in_speech:
            if r <= self.threshold:
                self.floor = 0.95 * self.floor + 0.05 * r
                self.loud = 0
                self.preroll.append(frame)
                return None
            self.loud += 1
            self.preroll.append(frame)
            if self.loud >= self.start_frames:
                self.in_speech, self.quiet, self.voiced = True, 0, self.loud
                self.buf = list(self.preroll)
                return "start"
            return None
        self.buf.append(frame)
        self.quiet = 0 if r > self.threshold else self.quiet + 1
        self.voiced += r > self.threshold
        if self.quiet >= self.end_frames or len(self.buf) >= self.max_frames:
            ok = self.voiced >= self.min_frames          # only frames that were actually loud count, not pre-roll
            self.in_speech = False
            self.loud = 0
            return "end" if ok else "noise"
        return None

    def audio(self):
        return np.concatenate(self.buf).astype(np.float32) if self.buf else np.zeros(0, np.float32)


# ----------------------------------------------------------------- hardware
class Microphone:
    def __init__(self, device=None, sr=SAMPLE_RATE, frame=FRAME):
        self.device, self.sr, self.frame, self.q, self.stream = device, sr, frame, queue.Queue(maxsize=400), None

    def open(self):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            raise SpeechError("sounddevice/PortAudio unavailable: " + str(e) + "  Fix: pip install sounddevice") from e
        try:
            if not [d for d in sd.query_devices() if d["max_input_channels"] > 0]:
                raise SpeechError("no microphone found. Plug in or enable one (Windows Settings > System > Sound > Input)")

            def cb(indata, frames, t, status):
                try:
                    self.q.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    pass  # drop frames rather than block the audio thread
            self.stream = sd.InputStream(samplerate=self.sr, channels=1, blocksize=self.frame, dtype="float32",
                                         device=self.device, callback=cb)
            self.stream.start()
        except SpeechError:
            raise
        except Exception as e:  # noqa: BLE001 - PortAudio raises many types
            raise SpeechError(f"could not open microphone: {e}. Check Windows microphone privacy settings.") from e
        return self

    def read(self, timeout=0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self):
        while not self.q.empty():
            self.q.get_nowait()

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class FasterWhisperSTT:
    def __init__(self, size="base.en", compute_type="int8", device="cpu", language="en", model_dir=None):
        self.size, self.compute_type, self.device, self.language, self.model_dir = size, compute_type, device, language, model_dir
        self.model = None

    def load(self):
        if self.model:
            return self
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise SpeechError("faster-whisper not installed. Fix: pip install faster-whisper") from e
        try:
            self.model = WhisperModel(self.model_dir or self.size, device=self.device, compute_type=self.compute_type)
        except Exception as e:  # noqa: BLE001 - network/disk errors vary
            raise SpeechError(f"could not load speech model {self.size!r}: {e}. The first run downloads it once "
                              f"(needs internet); afterwards it works offline. Or set LILI_STT_DIR to a local model folder.") from e
        return self

    def transcribe(self, audio):
        self.load()
        if audio is None or len(audio) < SAMPLE_RATE * 0.2:
            return ""
        segs, _ = self.model.transcribe(audio, language=self.language, beam_size=1, vad_filter=True,
                                        condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


_PS_SPEAK = r"""
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
[Console]::InputEncoding=[System.Text.Encoding]::UTF8
$text=[Console]::In.ReadToEnd()
$s=New-Object System.Speech.Synthesis.SpeechSynthesizer
$ok=$false
if($env:LILI_VOICE){ try{$s.SelectVoice($env:LILI_VOICE);$ok=$true}catch{ [Console]::Error.WriteLine('VOICE_NOT_FOUND') } }
if(-not $ok){ try{$s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female)}catch{} }
if($s.Voice.VoiceInfo.Gender -ne 'Female'){ [Console]::Error.WriteLine('NOT_FEMALE:'+$s.Voice.VoiceInfo.Name) }
$s.Rate=[int]$env:LILI_RATE
$s.Volume=[int]$env:LILI_VOLUME
$s.Speak($text)
"""
_PS_LIST = ("Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            ".GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Gender + '|' + $_.VoiceInfo.Culture }")
VOICE_HELP = ("Install a female voice: Windows Settings > Time & language > Speech > Manage voices > Add voices "
              "(e.g. an English voice with a female name such as 'Microsoft Zira' / 'Aria'). Then run: "
              "python main.py voice-test   and set LILI_VOICE to the exact name printed by `python main.py doctor`.")


class SapiTTS:
    """Windows' built-in offline voices (SAPI) through a PowerShell child process.
    The text goes in over stdin (never interpolated into a command). Killing the child interrupts speech instantly,
    which is what makes barge-in reliable. Voice name, rate and volume are configurable."""

    def __init__(self, voice=None, rate=0, volume=100):
        self.voice, self.rate, self.volume = voice, int(rate), int(volume)
        self.proc, self.last_error, self.launch_s, self.lock = None, "", 0.0, threading.Lock()

    @staticmethod
    def _ps():
        exe = shutil.which("powershell") or shutil.which("powershell.exe")
        if not exe:
            raise SpeechError("Windows PowerShell not found; SAPI voices need Windows. " + VOICE_HELP)
        return exe

    def list_voices(self):
        out = subprocess.run([self._ps(), "-NoProfile", "-NonInteractive", "-Command", _PS_LIST],
                             capture_output=True, text=True, timeout=30)
        if out.returncode:
            raise SpeechError("could not list voices: " + out.stderr.strip()[:200])
        return [tuple(l.strip().split("|")) for l in out.stdout.splitlines() if "|" in l]

    def check(self):
        try:
            voices = self.list_voices()
        except (SpeechError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        if self.voice and not any(v[0] == self.voice for v in voices):
            return False, f"voice {self.voice!r} not installed. Installed: {[v[0] for v in voices]}. " + VOICE_HELP
        if not self.voice and not any(v[1] == "Female" for v in voices):
            return False, f"no female voice installed (found {[v[0] for v in voices]}). " + VOICE_HELP
        return True, "voices: " + ", ".join(f"{n} ({g})" for n, g, *_ in voices)

    def speak(self, text):
        self.wait()
        text = (text or "").strip()
        if not text:
            return
        t0 = time.perf_counter()
        env = dict(os.environ, LILI_VOICE=self.voice or "", LILI_RATE=str(self.rate), LILI_VOLUME=str(self.volume))
        with self.lock:
            self.proc = subprocess.Popen([self._ps(), "-NoProfile", "-NonInteractive", "-Command", _PS_SPEAK], env=env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.proc.stdin.write(text.encode("utf-8"))
            self.proc.stdin.close()
        self.launch_s = time.perf_counter() - t0

    def is_speaking(self):
        if self.proc and self.proc.poll() is None:
            return True
        if self.proc:
            err = self.proc.stderr.read().decode(errors="replace").strip() if self.proc.stderr else ""
            self.last_error = err or ("exit code %d" % self.proc.returncode if self.proc.returncode else "")
            self.proc = None
        return False

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
            self.proc = None

    def wait(self, timeout=60):
        t0 = time.time()
        while self.is_speaking() and time.time() - t0 < timeout:
            time.sleep(0.02)


class PrintTTS:
    """Fallback when no voice is available: prints instead of speaking (text mode, no-audio machines)."""

    def __init__(self, out=print):
        self.out = out

    def check(self):
        return True, "text-only output"

    def speak(self, text):
        self.out("LILI: " + text)

    def is_speaking(self):
        return False

    def stop(self):
        pass

    def wait(self, timeout=0):
        pass


# ------------------------------------------------------------------ voice IO
class VoiceIO:
    """Mic -> VAD -> STT for listening; TTS with barge-in for speaking; echo filter between them."""

    def __init__(self, mic, stt, tts, barge_in=True, echo_window_s=6.0, vad_kwargs=None):
        self.mic, self.stt, self.tts, self.barge_in = mic, stt, tts, barge_in
        self.echo_window_s, self.vad_kwargs = echo_window_s, vad_kwargs or {}
        self.recent = deque(maxlen=5)          # (text, time spoken)
        self.carry = []                        # frames captured while interrupting, reused as pre-roll
        self.timings = {}                      # last measured latencies in seconds
        self._vad = EnergyVAD(**self.vad_kwargs)

    # ---- echo -----------------------------------------------------------
    def is_echo(self, text):
        t = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
        now = time.time()
        for spoken, ts in self.recent:
            if now - ts > self.echo_window_s + 0.1 * len(spoken.split()):
                continue
            s = re.sub(r"[^a-z0-9 ]", "", spoken.lower())
            if difflib.SequenceMatcher(None, t, s).ratio() > 0.75 or (len(t.split()) >= 3 and t in s):
                return True
        return False

    # ---- listening ------------------------------------------------------
    def listen(self, max_wait_s=30.0, cancel=None):
        """Block until one utterance is captured and transcribed. Returns text, or None (silence/unclear/cancel)."""
        vad, t_start = self._vad, time.time()  # persistent: an utterance may span several short listen() calls
        for f in self.carry:                   # audio captured during a barge-in belongs to this utterance
            vad.feed(f)
        self.carry = []
        while time.time() - t_start < max_wait_s:
            if cancel is not None and cancel.is_set():
                return None
            frame = self.mic.read(0.1)
            if frame is None:
                if getattr(self.mic, "exhausted", False):      # a scripted/file source ran dry
                    return None
                continue
            ev = vad.feed(frame)
            if ev == "end":
                t_end = time.time()
                self.timings["vad_endpoint_s"] = vad.end_frames * vad.fs          # inherent trailing-silence wait
                t0 = time.perf_counter()
                text = self.stt.transcribe(vad.audio())
                self.timings["stt_s"] = time.perf_counter() - t0
                self.timings["speech_end_at"] = t_end
                if not text or len(text.strip(" .,!?")) < 2:
                    vad.reset()
                    continue                                                       # unclear: keep listening
                if self.is_echo(text):
                    vad.reset()
                    continue                                                       # LILI heard herself
                return text
            if ev == "noise":
                vad.reset()
        return None

    # ---- speaking -------------------------------------------------------
    def say(self, text, block=True):
        """Speak. If the user starts talking over her, stop at once and return True (interrupted)."""
        self.tts.speak(text)
        self.recent.append((text, time.time()))
        self.timings["tts_launch_s"] = getattr(self.tts, "launch_s", 0.0)
        if not block:
            return False
        if not (self.barge_in and self.mic):
            self.tts.wait()
            return False
        vad = EnergyVAD(start_frames=8, mult=5.0, min_rms=0.02, **{k: v for k, v in self.vad_kwargs.items() if k not in ("start_frames", "mult", "min_rms")})
        self.mic.flush()
        while self.tts.is_speaking():
            frame = self.mic.read(0.03)
            if frame is None:
                continue
            if vad.feed(frame) == "start":
                self.tts.stop()
                self.carry = list(vad.buf)
                self.timings["barge_in_stop_at"] = time.time()
                return True
        self.recent[-1] = (text, time.time())
        return False
