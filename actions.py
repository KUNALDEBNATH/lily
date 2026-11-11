"""LILI actions: validated tools, a safety gate, execution with timeouts / limits / cancellation, and verification.

The model never gets to run code. It can only pick a tool from TOOLS with schema-valid arguments; the
SafetyGate then decides allow / ask-the-user / deny; the Executor checks preconditions, runs it, and the
Verifier confirms the outcome. Page text is untrusted: instruction-like text is stripped before the model
sees it, and even a fully hijacked model can only reach what the gate allows.
"""
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from model import validate
from perception import PerceptionUnavailable

KEYS = ["Enter", "Escape", "Tab", "PageDown", "PageUp", "ArrowDown", "ArrowUp", "Space", "End", "Home"]
_S = {"type": "string", "maxLength": 500}
TOOLS = {
    "browser_goto":   {"risk": "safe",  "schema": {"type": "object", "required": ["url"], "additionalProperties": False, "properties": {"url": {"type": "string", "maxLength": 2000}}}},
    "browser_click":  {"risk": "check", "needs_element": True, "schema": {"type": "object", "required": ["id"], "additionalProperties": False, "properties": {"id": {"type": "string", "maxLength": 20}}}},
    "browser_type":   {"risk": "check", "needs_element": True, "schema": {"type": "object", "required": ["id", "text"], "additionalProperties": False, "properties": {"id": {"type": "string", "maxLength": 20}, "text": {"type": "string", "maxLength": 2000}, "submit": {"type": "boolean"}}}},
    "browser_scroll": {"risk": "safe",  "schema": {"type": "object", "required": ["direction"], "additionalProperties": False, "properties": {"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer", "minimum": 1, "maximum": 10}}}},
    "browser_press":  {"risk": "check", "schema": {"type": "object", "required": ["key"], "additionalProperties": False, "properties": {"key": {"type": "string", "enum": KEYS}}}},
    "browser_wait":   {"risk": "safe",  "schema": {"type": "object", "required": ["seconds"], "additionalProperties": False, "properties": {"seconds": {"type": "number", "minimum": 0, "maximum": 10}}}},
    "browser_back":   {"risk": "safe",  "schema": {"type": "object", "additionalProperties": False, "properties": {}}},
    "browser_play":   {"risk": "safe",  "schema": {"type": "object", "additionalProperties": False, "properties": {}}},
    "read_page":      {"risk": "safe",  "schema": {"type": "object", "additionalProperties": False, "properties": {}}},
    "ui_click":       {"risk": "check", "needs_element": True, "schema": {"type": "object", "required": ["id"], "additionalProperties": False, "properties": {"id": {"type": "string", "maxLength": 20}}}},
    "ui_type":        {"risk": "check", "needs_element": True, "schema": {"type": "object", "required": ["id", "text"], "additionalProperties": False, "properties": {"id": {"type": "string", "maxLength": 20}, "text": {"type": "string", "maxLength": 2000}}}},
    "ui_hotkey":      {"risk": "check", "schema": {"type": "object", "required": ["keys"], "additionalProperties": False, "properties": {"keys": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 12}}}}},
    "ui_open_app":    {"risk": "safe",  "schema": {"type": "object", "required": ["name"], "additionalProperties": False, "properties": {"name": {"type": "string", "maxLength": 40}}}},
}
BROWSER_TOOLS = {t for t in TOOLS if t.startswith("browser_") or t == "read_page"}
READ_ONLY = {"browser_scroll", "browser_wait", "read_page", "browser_back", "browser_play"}
CONSEQUENTIAL = {"send", "delete", "remove", "buy", "purchase", "pay", "checkout", "order", "submit", "confirm",
                 "transfer", "post", "publish", "unsubscribe", "book", "withdraw", "erase", "format", "install"}
SEARCH_WORDS = {"search", "find", "query", "filter"}
BLOCKED_HOTKEYS = [{"alt", "f4"}, {"win", "r"}, {"ctrl", "alt", "delete"}, {"ctrl", "shift", "delete"}, {"ctrl", "shift", "escape"}]
DEFAULT_APPS = {"notepad": "notepad.exe", "calculator": "calc.exe", "paint": "mspaint.exe", "file explorer": "explorer.exe"}

_INJECTION = [re.compile(p, re.I) for p in [
    r"ignore (all |any |the )?(previous|prior|above|earlier)[^.\n]{0,40}(instructions|rules|prompt)",
    r"disregard[^.\n]{0,40}(instructions|rules|prompt)",
    r"new (system )?(instructions?|task|rules?)\s*:",
    r"(you|assistant|lili|ai)\s+(must|should|will)\s+now\b",
    r"do not (tell|inform|alert|notify) the user",
    r"(system|developer) prompt",
    r"\b(forward|send|email|upload|exfiltrate)\b[^.\n]{0,60}\b(password|credentials|cookies|token|inbox|contacts)\b",
    r"as an ai (language )?model[^.\n]{0,40}(you must|ignore)",
]]


def sanitize_untrusted(text):
    """Strip instruction-like sentences from page/app text. Returns (clean_text, [removed snippets])."""
    removed, out = [], []
    for sent in re.split(r"(?<=[.!?\n])\s+", text or ""):
        if any(p.search(sent) for p in _INJECTION):
            removed.append(sent.strip()[:120])
            out.append("[removed: instruction-like text found on the page]")
        else:
            out.append(sent)
    return " ".join(out), removed


# ---------------------------------------------------------------- safety gate
@dataclass
class Verdict:
    allow: bool
    confirm: bool = False
    reason: str = ""
    prompt: str = ""


@dataclass
class GateContext:
    user_text: str = ""
    injection_suspected: bool = False


class SafetyGate:
    def __init__(self, apps=None):
        self.apps = apps or dict(DEFAULT_APPS)

    @staticmethod
    def _consequential(*texts):
        words = set(re.findall(r"[a-z]+", " ".join(t for t in texts if t).lower()))
        return sorted(words & CONSEQUENTIAL)

    def check(self, tool, args, element=None, ctx=None):
        ctx = ctx or GateContext()
        spec = TOOLS.get(tool)
        if spec is None:
            return Verdict(False, reason=f"unknown tool {tool!r}")
        errs = validate(args if isinstance(args, dict) else {"_": args}, spec["schema"])
        if errs:
            return Verdict(False, reason="bad arguments: " + "; ".join(errs[:2]))
        reasons = []

        if tool == "browser_goto":
            u = urlparse(args["url"])
            if u.scheme not in ("http", "https") or not u.netloc:
                return Verdict(False, reason="only http(s) addresses are allowed")
            host = u.hostname or ""
            if ctx.injection_suspected and host.replace("www.", "") not in ctx.user_text.lower():
                return Verdict(False, reason="blocked navigation to a site you did not mention, after suspicious page text")
        if tool == "ui_open_app" and args["name"].lower() not in self.apps:
            return Verdict(False, reason=f"app {args['name']!r} is not on the allowed list {sorted(self.apps)}")
        if tool == "ui_hotkey":
            keys = {k.lower() for k in args["keys"]}
            if any(b <= keys for b in BLOCKED_HOTKEYS):
                return Verdict(False, reason="that shortcut is blocked")
            if "win" in keys:
                reasons.append("uses the Windows key")
        if element is not None:
            if element.state.get("password"):
                return Verdict(False, reason="I never type into password fields; please enter it yourself")
            if element.state.get("disabled"):
                return Verdict(False, reason="the target is disabled")
            hit = self._consequential(element.name, element.desc)
            if tool.endswith("click") and hit:
                reasons.append(f"'{element.name}' looks consequential ({', '.join(hit)})")
            if tool in ("browser_type", "ui_type"):
                from memory import contains_secret
                if contains_secret(args.get("text", "")):
                    reasons.append("the text looks like a password, key or card number")
                if args.get("submit") and not (set(re.findall(r"[a-z]+", (element.name + " " + element.desc).lower())) & SEARCH_WORDS):
                    reasons.append("it submits a form")
        if tool == "browser_press" and args["key"] == "Enter":
            reasons.append("pressing Enter may submit a form")
        if ctx.injection_suspected and tool not in READ_ONLY and not reasons:
            reasons.append("the page contained instruction-like text I removed")
        if reasons:
            return Verdict(True, True, "; ".join(reasons), "I need your OK: " + "; ".join(reasons) + ". Should I go ahead?")
        return Verdict(True)


# -------------------------------------------------------------------- adapters
class BrowserSession:
    """Owns the Playwright browser. A persistent profile means you sign in once, yourself; LILI never sees passwords."""

    def __init__(self, headless=False, profile_dir=None, channel=None):
        self.headless, self.profile_dir, self.channel = headless, profile_dir, channel
        self._pw = self._ctx = self.page = None

    def start(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise PerceptionUnavailable("playwright not installed. Fix: pip install playwright  then  python -m playwright install chromium") from e
        try:
            self._pw = sync_playwright().start()
            kw = {"headless": self.headless}
            if self.channel:
                kw["channel"] = self.channel
            if self.profile_dir:
                self._ctx = self._pw.chromium.launch_persistent_context(self.profile_dir, **kw)
            else:
                self._ctx = self._pw.chromium.launch(**kw).new_context()
            self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        except Exception as e:  # noqa: BLE001
            raise PerceptionUnavailable(f"could not start the browser: {e}. Fix: python -m playwright install chromium") from e
        return self

    def close(self):
        for o in (self._ctx, self._pw):
            try:
                o and (o.close() if hasattr(o, "close") else o.stop())
            except Exception:  # noqa: BLE001 - shutting down; nothing useful to do with a close error
                pass
        try:
            self._pw and self._pw.stop()
        except Exception:  # noqa: BLE001
            pass


class WindowsDesktop:
    """Mouse/keyboard/UIA actions on Windows. Uses UIA patterns first, coordinates only as a last resort.
    Moving the mouse into a screen corner aborts (pyautogui fail-safe) and is treated as 'stop'."""

    def __init__(self, apps=None):
        self.apps = apps or dict(DEFAULT_APPS)

    def _wrapper(self, handle):
        try:
            from pywinauto.controls.uiawrapper import UIAWrapper
        except ImportError as e:
            raise PerceptionUnavailable("pywinauto not installed. Fix: pip install pywinauto") from e
        return UIAWrapper(handle)

    def click(self, handle):
        w = self._wrapper(handle)
        try:
            w.invoke()
        except Exception:  # noqa: BLE001 - control does not support Invoke; fall back to a real click
            w.click_input()

    def type_text(self, handle, text):
        w = self._wrapper(handle)
        try:
            w.set_edit_text(text)
        except Exception:  # noqa: BLE001
            w.set_focus()
            w.type_keys(re.sub(r"([{}+^%~()])", r"{\1}", text), with_spaces=True)

    def hotkey(self, keys):
        import pyautogui
        pyautogui.hotkey(*[k.lower() for k in keys])

    def open_app(self, name):
        subprocess.Popen([self.apps[name.lower()]])


# -------------------------------------------------------------------- execution
@dataclass
class Result:
    ok: bool
    status: str                 # ok | denied | declined | precondition | stale | timeout | cancelled | limit | error
    detail: str = ""
    data: dict = field(default_factory=dict)
    latency_s: float = 0.0


class Executor:
    def __init__(self, perception, session=None, desktop=None, gate=None, confirm=None, cancel=None,
                 max_actions=25, timeout_s=15.0):
        self.perception, self.session, self.desktop = perception, session, desktop
        self.gate, self.confirm = gate or SafetyGate(), confirm          # confirm(prompt)->bool ; None == always decline
        self.cancel = cancel or threading.Event()
        self.max_actions, self.timeout_s = max_actions, timeout_s
        self.count, self.latencies, self.log = 0, [], []

    def reset_task(self):
        self.count = 0

    def stats(self):
        l = sorted(self.latencies)
        if not l:
            return {"n": 0}
        return {"n": len(l), "median_s": statistics.median(l), "p95_s": l[min(len(l) - 1, int(0.95 * len(l)))]}

    def _done(self, tool, args, res, t0):
        res.latency_s = time.perf_counter() - t0
        if res.status in ("ok", "error", "timeout"):
            self.latencies.append(res.latency_s)
        self.log.append((tool, args, res.status, res.detail))
        return res

    def run(self, tool, args, ctx=None, planned=None):
        """planned={'role','name'} is what the model saw when it chose the target; a mismatch means the page moved on."""
        t0 = time.perf_counter()
        args = args or {}
        if self.cancel.is_set():
            return self._done(tool, args, Result(False, "cancelled", "stopped by user"), t0)
        if self.count >= self.max_actions:
            return self._done(tool, args, Result(False, "limit", f"action limit ({self.max_actions}) reached"), t0)
        spec = TOOLS.get(tool)
        element = None
        if spec and spec.get("needs_element") and isinstance(args.get("id"), str):
            try:
                obs = self.perception.observe()
            except PerceptionUnavailable as e:
                return self._done(tool, args, Result(False, "precondition", str(e)), t0)
            element, status = self.perception.resolve({"id": args["id"]}, obs)
            if element is None:
                return self._done(tool, args, Result(False, "precondition", f"element {args['id']} is not on screen"), t0)
            if planned and ((planned.get("role") and planned["role"] != element.role) or
                            (planned.get("name") and planned["name"].strip().lower() != element.name.strip().lower())):
                return self._done(tool, args, Result(False, "stale", f"{args['id']} is now {element.brief()}, not what was planned"), t0)
        verdict = self.gate.check(tool, args, element, ctx)
        if not verdict.allow:
            return self._done(tool, args, Result(False, "denied", verdict.reason), t0)
        if verdict.confirm:
            if not (self.confirm and self.confirm(verdict.prompt)):
                return self._done(tool, args, Result(False, "declined", "you did not approve: " + verdict.reason), t0)
        self.count += 1
        try:
            if tool in BROWSER_TOOLS:                 # Playwright is thread-bound: rely on its own timeouts
                data = self._handler(tool, args, element)
            else:
                data = self._with_watchdog(lambda: self._handler(tool, args, element))
        except TimeoutError as e:
            return self._done(tool, args, Result(False, "timeout", str(e)), t0)
        except Exception as e:  # noqa: BLE001 - reported, never swallowed
            name = type(e).__name__
            if name == "FailSafeException":
                self.cancel.set()
                return self._done(tool, args, Result(False, "cancelled", "mouse fail-safe triggered"), t0)
            if "Timeout" in name:
                return self._done(tool, args, Result(False, "timeout", str(e)[:200]), t0)
            return self._done(tool, args, Result(False, "error", f"{name}: {str(e)[:200]}"), t0)
        finally:
            self.perception.invalidate(tool)
        return self._done(tool, args, Result(True, "ok", data=data or {}), t0)

    def _with_watchdog(self, fn):
        box = {}

        def go():
            try:
                box["v"] = fn()
            except BaseException as e:  # noqa: BLE001
                box["e"] = e
        t = threading.Thread(target=go, daemon=True)
        t.start()
        t.join(self.timeout_s)
        if t.is_alive():
            raise TimeoutError(f"action did not finish within {self.timeout_s:.0f}s")
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def _page(self):
        if not (self.session and self.session.page):
            raise PerceptionUnavailable("no browser is open")
        return self.session.page

    def _handler(self, tool, args, el):
        ms = int(self.timeout_s * 1000)
        if tool == "browser_goto":
            self._page().goto(args["url"], wait_until="domcontentloaded", timeout=ms)
        elif tool == "browser_click":
            self._page().locator(f'[data-lili-id="{el.id}"]').first.click(timeout=ms)
        elif tool == "browser_type":
            loc = self._page().locator(f'[data-lili-id="{el.id}"]').first
            loc.fill(args["text"], timeout=ms)
            if args.get("submit"):
                loc.press("Enter", timeout=ms)
        elif tool == "browser_scroll":
            self._page().mouse.wheel(0, (300 if args["direction"] == "down" else -300) * args.get("amount", 3))
        elif tool == "browser_press":
            self._page().keyboard.press(args["key"])
        elif tool == "browser_wait":
            time.sleep(min(float(args["seconds"]), self.timeout_s))
        elif tool == "browser_back":
            self._page().go_back(timeout=ms)
        elif tool == "browser_play":
            return {"playing": bool(self._page().evaluate("() => { const v=document.querySelector('video'); if(!v) return false; v.play(); return true; }"))}
        elif tool == "read_page":
            clean, removed = sanitize_untrusted(self.perception.observe(force=True).text)
            return {"text": clean, "removed": removed}
        elif tool == "ui_click":
            self._desk().click(el.handle)
        elif tool == "ui_type":
            self._desk().type_text(el.handle, args["text"])
        elif tool == "ui_hotkey":
            self._desk().hotkey(args["keys"])
        elif tool == "ui_open_app":
            self._desk().open_app(args["name"])
        else:
            raise ValueError("no handler for " + tool)
        return {}

    def _desk(self):
        if self.desktop is None:
            raise PerceptionUnavailable("desktop control is unavailable (Windows + pywinauto/pyautogui needed)")
        return self.desktop


# ------------------------------------------------------------------ verification
class Verifier:
    """Checks that an action really had its intended effect. No expectation => NOT verified."""

    def __init__(self, perception, session=None):
        self.perception, self.session = perception, session

    def check(self, expect, wait_s=5.0):
        if not expect:
            return False, "no expected outcome was given, so success cannot be confirmed"
        deadline, last = time.time() + wait_s, ""
        while True:
            ok, last = self._once(expect)
            if ok or time.time() >= deadline:
                return ok, last
            time.sleep(0.25)

    def _once(self, expect):
        known = {"url_contains", "url_not_contains", "title_contains", "text_contains", "element_contains", "media_playing"}
        bad = set(expect) - known
        if bad:
            return False, f"unknown expectation(s) {sorted(bad)}"
        try:
            obs = self.perception.observe(force=True)
        except PerceptionUnavailable as e:
            return False, f"cannot observe the screen: {e}"
        low = lambda s: (s or "").lower()
        if "url_contains" in expect and low(expect["url_contains"]) not in low(obs.url):
            return False, f"url {obs.url!r} lacks {expect['url_contains']!r}"
        if "url_not_contains" in expect and low(expect["url_not_contains"]) in low(obs.url):
            return False, f"url still contains {expect['url_not_contains']!r}"
        if "title_contains" in expect and low(expect["title_contains"]) not in low(obs.title):
            return False, f"title {obs.title!r} lacks {expect['title_contains']!r}"
        if "text_contains" in expect and low(expect["text_contains"]) not in low(obs.text):
            return False, f"page text lacks {expect['text_contains']!r}"
        if "element_contains" in expect and not any(low(expect["element_contains"]) in low(e.name) for e in obs.elements):
            return False, f"no element named like {expect['element_contains']!r}"
        if expect.get("media_playing"):
            try:
                playing = self.session.page.evaluate(
                    "() => { const v=document.querySelector('video'); return !!v && !v.paused && !v.ended && v.readyState>2; }")
            except Exception as e:  # noqa: BLE001
                return False, f"cannot inspect media: {e}"
            if not playing:
                return False, "no video is playing"
        return True, "expected outcome confirmed"
