"""LILI perception: understand the screen without sending screenshots to a vision model.

Priority (first source that yields elements wins):
  1. Browser DOM      (Playwright; re-read only when a MutationObserver says the page changed)
  2. Windows UIA      (pywinauto, read-only here; Windows only)
  3. OCR              (Tesseract on a screenshot; text only)
  4. OpenCV           (template matching for icons the others cannot name)
Everything returned is DATA about the screen. Page text is untrusted and is never an instruction.
"""
import hashlib
import platform
import re
import time
from dataclasses import dataclass, field

import numpy as np


class PerceptionUnavailable(Exception):
    pass


@dataclass
class Element:
    id: str
    source: str                 # dom | uia | ocr | cv
    role: str
    name: str
    state: dict = field(default_factory=dict)
    bbox: tuple = None          # x, y, w, h (viewport/screen pixels)
    ctx: str = ""               # surrounding text, used by the curiosity controller
    desc: str = ""              # title / aria-description
    handle: object = None       # opaque backend handle (UIA element); None for DOM (use id)

    def brief(self):
        st = ",".join(k for k, v in self.state.items() if v is True)
        return f'[{self.id}] {self.role} "{self.name[:60]}"' + (f" ({st})" if st else "")


@dataclass
class Observation:
    source: str
    elements: list
    url: str = ""
    title: str = ""
    text: str = ""              # visible page text: UNTRUSTED
    t: float = 0.0
    cached: bool = False

    @property
    def version(self):
        h = hashlib.sha1()
        for e in self.elements:
            h.update(f"{e.role}|{e.name}|{sorted(e.state.items())}".encode())
        h.update(self.url.encode())
        return h.hexdigest()[:12]


def summarize(obs, limit=40):
    head = f"source={obs.source} url={obs.url} title={obs.title!r}"
    return head + "\n" + "\n".join(e.brief() for e in obs.elements[:limit])


# ------------------------------------------------------------------ browser DOM
_JS_DIRTY = "() => window.__lili ? window.__lili.dirty : true"
_JS_EXTRACT = r"""
() => {
  if (!window.__lili) {
    window.__lili = {dirty: true, n: 0};
    window.__lili.obs = new MutationObserver(() => { window.__lili.dirty = true; });
    window.__lili.obs.observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
    window.addEventListener('scroll', () => { window.__lili.dirty = true; }, {passive: true});
  }
  const L = window.__lili;
  const SEL = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=tab],[role=menuitem],[role=checkbox],[onclick],h1,h2,h3';
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
                      return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const byIds = ids => ids.split(/\s+/).map(i => (document.getElementById(i) || {}).innerText || '').join(' ').trim();
  const nameOf = el => {
    const a = el.getAttribute('aria-label'); if (a) return a;
    const lb = el.getAttribute('aria-labelledby'); if (lb) { const t = byIds(lb); if (t) return t; }
    if (el.labels && el.labels.length) return Array.from(el.labels).map(l => l.innerText).join(' ').trim();
    const t = (el.innerText || '').trim(); if (t) return t;
    return el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('alt') ||
           (el.type === 'password' ? '' : el.value) || ((el.querySelector('img') || {}).alt) || '';
  };
  const roleOf = el => {
    const r = el.getAttribute('role'); if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'a') return 'link'; if (t === 'button' || t === 'summary') return 'button';
    if (t === 'select') return 'combobox'; if (t === 'textarea') return 'textbox';
    if (t === 'input') { const ty = (el.type || 'text').toLowerCase();
      return ({checkbox: 'checkbox', radio: 'radio', button: 'button', submit: 'button', range: 'slider'})[ty] || 'textbox'; }
    if (/^h[1-3]$/.test(t)) return 'heading';
    return 'generic';
  };
  const out = [];
  document.querySelectorAll(SEL).forEach(el => {
    if (!vis(el)) return;
    let id = el.getAttribute('data-lili-id');
    if (!id) { id = 'e' + (++L.n); el.setAttribute('data-lili-id', id); }
    const r = el.getBoundingClientRect();
    const host = el.closest('li,article,section,tr,form,[role=listitem]');
    out.push({id, role: roleOf(el), name: nameOf(el).replace(/\s+/g, ' ').slice(0, 120),
      desc: (el.getAttribute('title') || el.getAttribute('aria-description') || '').slice(0, 160),
      ctx: host ? (host.innerText || '').replace(/\s+/g, ' ').slice(0, 160) : '',
      state: {disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
              checked: !!el.checked || el.getAttribute('aria-checked') === 'true',
              expanded: el.getAttribute('aria-expanded') === 'true',
              password: el.type === 'password',
              href: !!el.getAttribute('href')},
      href: el.getAttribute('href') || '',
      bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]});
  });
  L.obs.takeRecords();      // our own data-lili-id writes must not mark the page dirty
  L.dirty = false;
  return {url: location.href, title: document.title, text: (document.body ? document.body.innerText : '').slice(0, 4000), elements: out};
}
"""


class BrowserPerceiver:
    def __init__(self, page):
        self.page, self.last, self.extractions, self.reuses = page, None, 0, 0

    def observe(self, force=False):
        try:
            if not force and self.last is not None:
                if self.page.url == self.last.url and not self.page.evaluate(_JS_DIRTY):
                    self.reuses += 1
                    self.last.cached = True
                    return self.last
            try:
                data = self.page.evaluate(_JS_EXTRACT)
            except Exception:  # noqa: BLE001 - navigation can destroy the context mid-call; retry once
                self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                data = self.page.evaluate(_JS_EXTRACT)
        except Exception as e:  # noqa: BLE001
            raise PerceptionUnavailable(f"browser page not readable: {e}") from e
        self.extractions += 1
        els = []
        for d in data["elements"]:
            st = dict(d["state"])
            st["href_value"] = d.get("href", "")
            els.append(Element(d["id"], "dom", d["role"], d["name"], st, tuple(d["bbox"]), d["ctx"], d["desc"]))
        self.last = Observation("dom", els, data["url"], data["title"], data["text"], time.monotonic())
        return self.last


# ---------------------------------------------------------------------- UIA
_UIA_TYPES = {"Button": "button", "Edit": "textbox", "MenuItem": "menuitem", "ListItem": "listitem",
              "Hyperlink": "link", "CheckBox": "checkbox", "ComboBox": "combobox", "TabItem": "tab",
              "RadioButton": "radio", "Text": "text", "Document": "document", "SplitButton": "button"}


class UIAReader:
    """Read-only Windows UI Automation via pywinauto. Not testable off-Windows; the tests use a mock with the
    same controls() interface, and `python main.py doctor` reports whether this one works on your machine."""

    def controls(self, max_controls=200):
        if platform.system() != "Windows":
            raise PerceptionUnavailable("Windows UI Automation is only available on Windows")
        try:
            import ctypes
            from pywinauto import Desktop
        except ImportError as e:
            raise PerceptionUnavailable("pywinauto not installed. Fix: pip install pywinauto") from e
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            win = Desktop(backend="uia").window(handle=hwnd)
            out = []
            for info in [c.element_info for c in win.descendants()][:max_controls * 3]:
                ctype = info.control_type
                if ctype not in _UIA_TYPES or not info.visible:
                    continue
                r = info.rectangle
                out.append({"control_type": ctype, "name": info.name or "", "enabled": bool(info.enabled),
                            "rect": (r.left, r.top, r.right - r.left, r.bottom - r.top),
                            "automation_id": info.automation_id or "", "handle": info})
                if len(out) >= max_controls:
                    break
            return out, win.window_text()
        except Exception as e:  # noqa: BLE001
            raise PerceptionUnavailable(f"UI Automation read failed: {e}") from e


class UIAPerceiver:
    def __init__(self, reader=None):
        self.reader = reader or UIAReader()

    def observe(self):
        res = self.reader.controls()
        controls, title = res if isinstance(res, tuple) else (res, "")
        els = [Element(f"u{i}", "uia", _UIA_TYPES.get(c["control_type"], c["control_type"].lower()), c["name"],
                       {"disabled": not c.get("enabled", True)}, tuple(c.get("rect") or ()) or None,
                       desc=c.get("automation_id", ""), handle=c.get("handle")) for i, c in enumerate(controls)]
        return Observation("uia", els, "", title, " ".join(e.name for e in els if e.role == "text")[:3000], time.monotonic())


# ------------------------------------------------------------ OCR + OpenCV
def grab_screen():
    try:
        from PIL import ImageGrab
        return np.array(ImageGrab.grab().convert("RGB"))
    except Exception as e:  # noqa: BLE001
        raise PerceptionUnavailable(f"screen capture unavailable: {e}. Fix: pip install pillow") from e


def ocr_elements(image, region=None, min_conf=45, lang="eng"):
    """OCR an image (numpy array) or a region (x, y, w, h) of it -> text-line Elements with boxes."""
    try:
        import pytesseract
    except ImportError as e:
        raise PerceptionUnavailable("pytesseract not installed. Fix: pip install pytesseract") from e
    ox = oy = 0
    if region:
        x, y, w, h = region
        image, ox, oy = image[y:y + h, x:x + w], x, y
    try:
        d = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise PerceptionUnavailable("Tesseract program not found. Windows: install from "
                                    "https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH") from e
    lines = {}
    for i, word in enumerate(d["text"]):
        if not word.strip() or float(d["conf"][i]) < min_conf:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append((word, d["left"][i], d["top"][i], d["width"][i], d["height"][i]))
    out = []
    for n, ws in enumerate(lines.values()):
        x0, y0 = min(w[1] for w in ws), min(w[2] for w in ws)
        x1, y1 = max(w[1] + w[3] for w in ws), max(w[2] + w[4] for w in ws)
        out.append(Element(f"o{n}", "ocr", "text", " ".join(w[0] for w in ws), {}, (x0 + ox, y0 + oy, x1 - x0, y1 - y0)))
    return out


def find_template(screen, template, threshold=0.85, label="icon"):
    """Lightweight icon matching with OpenCV. Returns Elements, best first (non-maximum suppressed)."""
    try:
        import cv2
    except ImportError as e:
        raise PerceptionUnavailable("opencv not installed. Fix: pip install opencv-python-headless") from e
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im
    s, t = g(screen), g(template)
    if t.shape[0] > s.shape[0] or t.shape[1] > s.shape[1]:
        return []
    res = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    cands = sorted(zip(res[ys, xs], xs, ys), reverse=True)
    h, w = t.shape
    picked = []
    for score, x, y in cands:
        if all(abs(x - px) > w // 2 or abs(y - py) > h // 2 for _, px, py in picked):
            picked.append((score, int(x), int(y)))
    return [Element(f"c{i}", "cv", label, label, {"score": float(sc)}, (x, y, w, h)) for i, (sc, x, y) in enumerate(picked)]


# --------------------------------------------------------------------- facade
class Perception:
    def __init__(self, browser=None, uia=None, ttl=2.0, clock=time.monotonic, ocr_fallback=True):
        self.browser, self.uia, self.ttl, self.clock, self.ocr_fallback = browser, uia, ttl, clock, ocr_fallback
        self._cache, self._cache_t, self.invalidations, self.reasons = None, 0.0, 0, []

    def attach_browser(self, page):
        self.browser = BrowserPerceiver(page)
        self.invalidate("browser attached")

    def invalidate(self, reason=""):
        """Call after any action that could change the screen. The DOM source also self-detects changes."""
        self._cache = None
        self.invalidations += 1

    def observe(self, force=False):
        self.reasons = []
        if self.browser is not None:
            try:
                obs = self.browser.observe(force=force or self._cache is None)
                if obs.elements or obs.text or obs.title:       # a text-only page is still a valid, verifiable observation
                    self._cache, self._cache_t = obs, self.clock()
                    return obs
                self.reasons.append("browser page was empty")
            except PerceptionUnavailable as e:
                self.reasons.append(str(e))
        if not force and self._cache is not None and self._cache.source != "dom" and self.clock() - self._cache_t < self.ttl:
            self._cache.cached = True
            return self._cache
        if self.uia is not None:
            try:
                obs = self.uia.observe()
                if obs.elements:
                    self._cache, self._cache_t = obs, self.clock()
                    return obs
                self.reasons.append("UI Automation returned no controls")
            except PerceptionUnavailable as e:
                self.reasons.append(str(e))
        if self.ocr_fallback:
            try:
                els = ocr_elements(grab_screen())
                obs = Observation("ocr", els, "", "", " ".join(e.name for e in els)[:3000], self.clock())
                self._cache, self._cache_t = obs, self.clock()
                return obs
            except PerceptionUnavailable as e:
                self.reasons.append(str(e))
        raise PerceptionUnavailable("no perception source worked: " + " | ".join(self.reasons))

    # --- semantic selection ---------------------------------------------
    @staticmethod
    def _tok(s):
        return set(re.findall(r"[a-z0-9]{2,}", (s or "").lower()))

    def find(self, query, obs=None, role=None, limit=5):
        """Rank elements for a text query. Returns [(score, Element)] best first."""
        obs = obs or self._cache or self.observe()
        q = self._tok(query)
        if not q:
            return []
        scored = []
        for e in obs.elements:
            n = self._tok(e.name) | self._tok(e.desc)
            s = len(q & n) / len(q) * 0.8 + (0.4 if (e.name or "").lower().strip() == query.lower().strip() else 0.0)
            s += 0.1 * len(q & self._tok(e.ctx)) / len(q) + (0.1 if role and e.role == role else 0.0)
            if role and e.role != role:
                s -= 0.3
            if s >= 0.35:
                scored.append((round(s, 3), e))
        return sorted(scored, key=lambda t: -t[0])[:limit]

    def resolve(self, target, obs=None):
        """target: {"id"?, "role"?, "name"?}. Returns (element|None, status) with status ok|missing|ambiguous."""
        obs = obs or self._cache or self.observe()
        if target.get("id"):
            for e in obs.elements:
                if e.id == target["id"]:
                    return e, "ok"
            return None, "missing"
        ranked = self.find(target.get("name", ""), obs, target.get("role"))
        if not ranked:
            return None, "missing"
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.15 and ranked[1][0] >= 0.5:
            return ranked[0][1], "ambiguous"
        return ranked[0][1], "ok"
