"""LILI learning.

Level 1  immediate memory   : curiosity questions, answers and corrections are stored at once (no training).
Level 2  reusable skills    : a skill is created/promoted ONLY from a task whose outcome was verified.
Level 3  model weights      : optional LoRA candidate training behind data validation, a regression gate,
                              versioning and rollback. Heavy dependencies are imported only inside train_candidate().
Memory learning and weight learning are separate on purpose: a fact is learned the moment you say it;
weights change only when a candidate beats the current model on held-out and regression cases.
"""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field

from actions import CONSEQUENTIAL, TOOLS
from memory import contains_secret, norm, redact, sig, tokens
from model import decision_schema, parse_json_loose, validate

COMMON_CONTROLS = {"ok", "cancel", "close", "back", "next", "previous", "search", "menu", "home", "help", "settings",
                   "log in", "login", "sign in", "more", "yes", "no", "open", "save", "play", "pause", "share", "like",
                   "continue", "skip", "done", "apply", "refresh", "download", "copy", "paste", "undo", "redo", "preferences"}
COMMON_WORDS = {w for c in COMMON_CONTROLS for w in c.split()} | {"the", "a", "to", "of", "all", "now", "new", "page", "item", "message"}
UNIVERSAL = {"send", "delete", "submit", "remove", "buy", "pay", "checkout", "post", "publish", "order", "confirm", "save"}
HIGH_CONSEQUENCE = CONSEQUENTIAL |{"archive", "spam", "block", "report", "mute", "reset", "clear", "revoke", "disable",
                                    "discard", "empty", "trash", "overwrite", "terminate", "kill", "wipe", "move"}
KNOWN_APPS = {"gmail", "youtube", "outlook", "chrome", "edge", "excel", "word", "powerpoint", "notepad", "explorer",
              "slack", "teams", "zoom", "whatsapp", "drive", "docs", "sheets", "github", "spotify"}


# ===================================================================== curiosity
@dataclass
class Unknown:
    concept: str
    app: str
    role: str
    consequence: str            # low | high
    hypothesis: str = ""
    hypothesis_source: str = ""  # interface | web
    decision: str = "ask"       # known | ignore | proceed | ask
    question: str = ""


class Curiosity:
    """Observe -> search memory -> weigh consequences -> investigate -> ask only if it still matters."""

    def __init__(self, memory, search_fn=None):
        self.mem, self.search_fn, self.skipped = memory, search_fn, set()   # skipped: asked once, user said skip

    def evaluate(self, element, instruction="", app="", relevant_only=True):
        concept = norm(element.name)
        toks = tokens(concept)
        u = Unknown(concept, norm(app), element.role, "low")
        if not toks or len(concept) > 40:
            u.decision = "ignore"
            return u
        facts = self.mem.get_facts(concept, app)
        if (concept, norm(app)) in self.skipped or (facts and facts[0]["status"] == "user_confirmed"):
            u.decision = "known"
            return u
        if toks & UNIVERSAL or concept in COMMON_CONTROLS or toks <= COMMON_WORDS:
            u.decision = "ignore"                      # meaning is universal; the safety gate still guards consequences
            return u
        u.consequence = "high" if toks & HIGH_CONSEQUENCE else "low"
        if facts and u.consequence == "low":
            u.decision = "known"                       # a guess is good enough for harmless controls...
            return u
        relevant = bool(toks & tokens(instruction))
        if relevant_only and not relevant:
            u.decision = "ignore"                      # do not quiz the user about every button
            return u
        if facts:                                      # ...but for consequential ones only the user can confirm it
            u.hypothesis, u.hypothesis_source = facts[0]["meaning"], "memory"
        # investigate: tooltip / accessible description, then (optionally) external search
        if u.hypothesis:
            pass
        elif element.desc:
            u.hypothesis, u.hypothesis_source = element.desc, "interface"
        elif self.search_fn:
            found = self.search_fn(concept, app)
            if found:
                u.hypothesis, u.hypothesis_source = found[0], "web"
        if u.hypothesis and u.consequence == "low":
            self.mem.add_fact(concept, u.hypothesis, app, source=u.hypothesis_source, confidence=0.6,
                              context=element.ctx[:120], retrieved_at=time.strftime("%Y-%m-%d") if u.hypothesis_source == "web" else None)
            u.decision = "proceed"                     # stored as INFERRED, never user_confirmed
            return u
        where = f" in {app}" if app else ""
        h = u.hypothesis.strip().rstrip(".")
        if not h:
            tail = "What does it do?"
        elif h.split()[0].lower().endswith("s") and not h.split()[0].lower().endswith("ss"):   # "Moves the item..." -> a verb
            tail = f"I think it {h[0].lower() + h[1:]}. Is that right?"
        else:
            tail = f"Its description says '{h}'. Is that what it does?"
        u.question = f"I found a {element.role} labelled '{element.name}'{where}. " + tail
        u.decision = "ask"
        return u

    def assess(self, obs, instruction="", app="", max_questions=1, relevant_only=True):
        """Questions worth asking about this screen (at most `max_questions`, highest consequence first)."""
        asks = []
        for e in obs.elements:
            is_control = e.role in ("button", "menuitem", "tab", "checkbox") or (e.role == "link" and len(e.name.split()) <= 2)
            if is_control:                                     # content links ("Message from ...") are not controls
                u = self.evaluate(e, instruction, app, relevant_only)
                if u.decision == "ask":
                    asks.append(u)
        asks.sort(key=lambda u: u.consequence != "high")
        return asks[:max_questions]

    def receive_answer(self, u, answer):
        """Turn the user's reply into memory. Returns {'status': stored|need_more|skipped, ...}."""
        a = answer.strip()
        low = norm(a)
        if re.match(r"^(skip|never ?mind|i don'?t know|not sure|ignore (it|that))\b", low):
            self.skipped.add((u.concept, u.app))
            return {"status": "skipped"}
        yes = re.match(r"^(yes|yeah|yep|yup|correct|right|exactly|that'?s (right|correct)|sure)\b", low)
        no = re.match(r"^(no|nope|not quite|wrong|incorrect|that'?s (wrong|not right))\b", low)
        if yes and u.hypothesis:
            fid = self.mem.add_fact(u.concept, u.hypothesis, u.app, user_confirmed=True)
            return {"status": "stored", "fact_id": fid, "meaning": u.hypothesis}
        if no:
            rest = re.sub(r"^(no|nope|not quite|wrong|incorrect|that'?s (wrong|not right))[\s,.:;-]*", "", a, flags=re.I).strip()
            if len(rest.split()) < 3:
                return {"status": "need_more", "question": f"Then what does '{u.concept}' do?"}
            a = rest
        elif yes:
            return {"status": "need_more", "question": f"Could you tell me in a few words what '{u.concept}' does?"}
        fid = self.mem.add_fact(u.concept, a, u.app, user_confirmed=True)
        return {"status": "stored", "fact_id": fid, "meaning": a}


# ==================================================================== corrections
_TEMP = re.compile(r"\b(this time|just this once|for now|right now|only (for )?(this|today)|for this (task|one|time)|today only)\b", re.I)
_PREF = re.compile(r"\b(always|never|from now on|whenever|every time|don'?t ever|make sure (you|to)|i prefer|i (like|want) you to|automatically)\b", re.I)
_SKILL = re.compile(r"\b(the (right|correct|better|proper) way|use this method|use this approach|instead of|do it (by|like)|first .{3,60}\bthen\b|steps? (are|is))\b", re.I)
_FACT = re.compile(r"\bmeans\b|\bis (a|the) (button|icon|link|menu|tab)\b|\bis used (to|for)\b|\b(button|icon|menu|tab|link)\b.{0,30}\b(does|moves|opens|closes|shows|deletes|saves)\b", re.I)
_DEICTIC = re.compile(r"\b(that|this|those|these|it|the other)\b", re.I)
_NEG_START = re.compile(r"^(no\b|not\b|wrong\b|don'?t\b|do not\b|stop\b|that was wrong|you (should not|shouldn'?t))", re.I)


def classify_correction(text, known_apps=()):
    """Decide what kind of lesson this is, so a one-off remark never becomes a permanent rule.
    Returns {'scope': temporary|skill|app_fact|preference|action, 'app': str, 'concept'?, 'meaning'?}."""
    t = text.strip()
    apps = set(KNOWN_APPS) | {norm(a) for a in known_apps}
    app = next((a for a in sorted(apps) if re.search(rf"\b{re.escape(a)}\b", t, re.I)), "")
    out = {"scope": "preference", "app": app, "text": t}
    if _TEMP.search(t):
        out["scope"] = "temporary"
    elif _SKILL.search(t):
        out["scope"] = "skill"
    elif _FACT.search(t):
        out["scope"] = "app_fact"
        m = re.search(r"(?:the\s+)?['\"]?([\w ]{1,30}?)['\"]?\s+(?:button|icon|menu|tab|link)?\s*(?:means|is used to|is used for|is for|does|moves|opens|closes)\s+(.+)", t, re.I)
        if m:
            out["concept"], out["meaning"] = m.group(1).strip(), m.group(2).strip(" .")
        else:
            out["scope"] = "preference"
    elif _PREF.search(t):
        out["scope"] = "preference"
    elif _NEG_START.search(t) and _DEICTIC.search(t):
        out["scope"] = "action"
    elif _NEG_START.search(t):
        out["scope"] = "preference"                    # "Do not open the first result" with no specific target
    else:
        out["scope"] = "action"
    return out


def apply_correction(memory, text, session_notes, known_apps=()):
    """Store a correction in the right place. Returns (classification, message for LILI to say)."""
    c = classify_correction(text, known_apps)
    if contains_secret(text):
        return c, "That looks like it contains a password or key, so I won't store it."
    if c["scope"] == "temporary":
        session_notes.append(text)
        return c, "Okay, I'll do that for this task only."
    if c["scope"] == "app_fact" and c.get("concept"):
        memory.add_fact(c["concept"], c["meaning"], c["app"], user_confirmed=True)
        return c, f"Got it. '{c['concept']}' means: {c['meaning']}. I'll remember that."
    memory.add_correction(text, c["scope"], c["app"])
    kind = {"preference": "a standing preference", "action": "a correction for that action", "skill": "a better way to do it",
            "app_fact": "a note about the app"}[c["scope"]]
    return c, f"Understood. I've saved that as {kind}."


# ============================================================== experience -> skills
@dataclass
class TaskRecord:
    instruction: str
    steps: list = field(default_factory=list)      # {tool,args,target,ok,view?}
    outcome: str = "unverified"                    # success | failure | unverified | cancelled
    verified: bool = False
    expect: dict = field(default_factory=dict)
    skill_used: str = ""
    app: str = ""
    failure: dict = field(default_factory=dict)    # {step, context, reason}
    model_calls: int = 0


def extract_params(instruction):
    """'open youtube and find a good ai tutorial' -> ('open youtube and find {query}', {'query': 'a good ai tutorial'})."""
    m = re.search(r"\b(?:find|search(?: for)?|look up|look for)\s+(.+?)\s*[.?!]?$", instruction.strip(), re.I)
    if m:
        return instruction[:m.start(1)] + "{query}", {"query": m.group(1)}
    return instruction.strip(), {}


def parametrize(steps, params):
    s = json.dumps(steps)
    for k, v in params.items():
        if v:
            s = s.replace(json.dumps(v)[1:-1], "{" + k + "}")
    return json.loads(s)


def fill(steps, params):
    s = json.dumps(steps)
    for k, v in params.items():
        s = s.replace("{" + k + "}", json.dumps(v)[1:-1])
    return json.loads(s)


class Learner:
    def __init__(self, memory):
        self.mem = memory

    def learn(self, rec: TaskRecord):
        """Level 1/2 learning from one finished task. Returns a summary dict (also useful for tests/benchmarks)."""
        steps_for_log = [{"tool": s["tool"], "args": s.get("args", {}), "target": s.get("target"), "ok": s.get("ok", True),
                          "view": s.get("view", [])} for s in rec.steps]
        lesson = ""
        if rec.outcome == "failure":
            lesson = f"failed at {rec.failure.get('step', '?')}: {rec.failure.get('reason', 'unknown')}"
        elif rec.outcome == "unverified":
            lesson = "finished but the expected outcome was not confirmed"
        self.mem.add_episode(rec.instruction, steps_for_log, rec.outcome, lesson)
        summary = {"skill": None, "promoted": False, "failure_recorded": False}

        if rec.outcome == "failure":
            f = rec.failure
            self.mem.record_failure(rec.instruction, f.get("step", "unknown"), f.get("context", ""), f.get("reason", ""))
            summary["failure_recorded"] = True
            if rec.skill_used:
                self.mem.record_skill_run(rec.skill_used, success=False, verified=False)
            return summary

        if rec.outcome == "success" and rec.verified:
            if rec.skill_used:
                s = self.mem.record_skill_run(rec.skill_used, success=True, verified=True)
                summary.update(skill=rec.skill_used, promoted=bool(s and s["status"] == "verified"))
                return summary
            usable = [s for s in rec.steps if s.get("ok", True) and s["tool"] not in ("read_page", "browser_wait")]
            if usable and all((TOOLS[s["tool"]].get("needs_element") is None) or s.get("target") for s in usable if s["tool"] in TOOLS):
                template, params = extract_params(rec.instruction)
                body = parametrize([{"tool": s["tool"], "args": {k: v for k, v in s.get("args", {}).items() if k != "id"},
                                     "target": s.get("target")} for s in usable], params)
                name = re.sub(r"[^a-z0-9]+", "_", re.sub(r"\{[a-z_]+\}", "", template.lower())).strip("_")[:50] or "skill"
                self.mem.save_skill(name, template, body, rec.expect, params=params)
                s = self.mem.record_skill_run(name, success=True, verified=True)
                summary.update(skill=name, promoted=bool(s and s["status"] == "verified"))
        return summary                                 # unverified/cancelled => episode only, never a skill


# ============================================================ training data (Level 3)
def system_prompt_for_training(rules):
    base = "You are LILI, a careful computer-use agent. Reply with one JSON decision."
    return base + (("\nRules from the user:\n- " + "\n- ".join(rules)) if rules else "")


def build_examples(memory):
    """Per-step SFT examples from VERIFIED successful episodes. The user's corrections that apply to a task become
    rules in that example's system prompt. Failed episodes are not imitated (we never train on a wrong action)."""
    out = []
    for ep in memory.episodes(outcome="success", limit=5000):
        rules = [c["text"] for c in memory.corrections_for(ep["instruction"])]
        done = []
        for st in ep["steps"]:
            if not st.get("view") or not st.get("ok", True):       # only steps with a recorded view are learnable
                done.append(st["tool"])
                continue
            user = f"Task: {ep['instruction']}\nDone so far: {', '.join(done) or 'nothing'}\nScreen:\n" + "\n".join(st["view"])
            out.append({"messages": [{"role": "system", "content": system_prompt_for_training(rules)},
                                     {"role": "user", "content": user},
                                     {"role": "assistant", "content": json.dumps({"action": st["tool"], "args": st["args"]})}],
                        "source": "episode"})
            done.append(st["tool"])
    return out


_DECISION = decision_schema(TOOLS)


def validate_example(ex):
    """Returns '' if valid else the reason it was rejected."""
    try:
        msgs = ex["messages"]
        if [m["role"] for m in msgs] != ["system", "user", "assistant"]:
            return "bad roles"
        text = " ".join(m["content"] for m in msgs)
        if not text.strip() or len(text) > 6000:
            return "empty or too long"
        if contains_secret(text) or redact(text) != text:
            return "contains a secret"
        d = parse_json_loose(msgs[2]["content"])
        errs = validate(d, _DECISION)
        if errs:
            return "assistant output invalid: " + errs[0]
        if d["action"] in TOOLS:
            errs = validate(d.get("args", {}), TOOLS[d["action"]]["schema"])
            if errs:
                return "tool args invalid: " + errs[0]
    except (KeyError, TypeError, ValueError) as e:
        return f"malformed: {e}"
    return ""


def dedupe(examples):
    seen, out = set(), []
    for ex in examples:
        k = sig(*(m["content"] for m in ex["messages"]))
        if k not in seen:
            seen.add(k)
            out.append(ex)
    return out


def split(examples, eval_every=5):
    """Deterministic split by content hash, so an example can never drift between train and eval across exports."""
    train, ev = [], []
    for ex in examples:
        h = int(hashlib.sha1(sig(*(m["content"] for m in ex["messages"])).encode()).hexdigest(), 16)
        (ev if h % eval_every == 0 else train).append(ex)
    if not ev and len(train) >= eval_every:                       # tiny sets: guarantee a held-out example
        ev.append(train.pop())
    return train, ev


def export_training_data(memory, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = build_examples(memory)
    ok, rejected = [], {}
    for ex in raw:
        why = validate_example(ex)
        if why:
            rejected[why] = rejected.get(why, 0) + 1
        else:
            ok.append(ex)
    uniq = dedupe(ok)
    train, ev = split(uniq)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", ev)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")
    manifest = {"raw": len(raw), "valid": len(ok), "duplicates_removed": len(ok) - len(uniq), "train": len(train),
                "eval": len(ev), "rejected": rejected, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ================================================ versioning, evaluation, rollback
class ModelRegistry:
    """JSON file of model versions. Deploying requires a passed evaluation; the previous model is always kept."""

    def __init__(self, path, base_model="qwen3:4b"):
        self.path = path
        if os.path.exists(path):
            with open(path) as f:
                self.d = json.load(f)
        else:
            self.d = {"current": "v0", "previous": None,
                      "versions": {"v0": {"model": base_model, "status": "deployed", "metrics": {}, "created": time.time()}}}
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.d, f, indent=2)

    def current(self):
        return self.d["versions"][self.d["current"]]["model"]

    def register_candidate(self, model_name, metrics):
        v = f"v{len(self.d['versions'])}"
        self.d["versions"][v] = {"model": model_name, "status": "candidate", "metrics": metrics, "created": time.time()}
        self._save()
        return v

    def deploy(self, version):
        ver = self.d["versions"][version]
        if not ver["metrics"].get("passed"):
            ver["status"] = "rejected"
            self._save()
            raise ValueError(f"{version} did not pass evaluation and cannot be deployed")
        self.d["versions"][self.d["current"]]["status"] = "retired"
        self.d["previous"], self.d["current"], ver["status"] = self.d["current"], version, "deployed"
        self._save()

    def rollback(self):
        if not self.d["previous"]:
            raise ValueError("no previous model to roll back to")
        bad = self.d["current"]
        self.d["versions"][bad]["status"] = "rolled_back"
        self.d["current"] = self.d["previous"]
        self.d["versions"][self.d["current"]]["status"] = "deployed"
        self.d["previous"] = None
        self._save()


GOLDEN = [  # regression cases: tiny, deterministic, must never get worse
    ([{"role": "system", "content": system_prompt_for_training([])},
      {"role": "user", "content": "Task: scroll down\nDone so far: nothing\nScreen:\n[e1] link \"Home\""}], "browser_scroll"),
    ([{"role": "system", "content": system_prompt_for_training([])},
      {"role": "user", "content": "Task: go to https://example.com\nDone so far: nothing\nScreen:\n"}], "browser_goto"),
]


def score_model(generate, cases):
    """cases: [(messages, expected_action)]. generate(messages)->text. Returns accuracy, valid_rate, per-case hits."""
    hits, valid = [], 0
    for msgs, expected in cases:
        try:
            d = parse_json_loose(generate(msgs))
            ok_schema = not validate(d, _DECISION)
        except Exception:  # noqa: BLE001 - a crash or bad JSON simply scores zero for that case
            d, ok_schema = {}, False
        valid += ok_schema
        hits.append(bool(ok_schema and d.get("action") == expected))
    n = max(1, len(cases))
    return {"accuracy": sum(hits) / n, "valid_rate": valid / n, "hits": hits}


def evaluate_candidate(baseline_gen, candidate_gen, eval_examples, golden=GOLDEN, min_gain=0.0):
    """Pass only if: no golden regression, valid-JSON rate not lower, accuracy >= baseline + min_gain."""
    held = [(ex["messages"][:2], parse_json_loose(ex["messages"][2]["content"])["action"]) for ex in eval_examples]
    b_gold, c_gold = score_model(baseline_gen, golden), score_model(candidate_gen, golden)
    b_held, c_held = score_model(baseline_gen, held), score_model(candidate_gen, held)
    regress = [i for i, (b, c) in enumerate(zip(b_gold["hits"], c_gold["hits"])) if b and not c]
    reasons = []
    if regress:
        reasons.append(f"regressed on golden cases {regress}")
    if c_held["valid_rate"] < b_held["valid_rate"]:
        reasons.append("valid-JSON rate dropped")
    if held and c_held["accuracy"] < b_held["accuracy"] + min_gain:
        reasons.append(f"held-out accuracy {c_held['accuracy']:.2f} < baseline {b_held['accuracy']:.2f} + {min_gain}")
    if not held:
        reasons.append("no held-out examples to evaluate on")
    return {"passed": not reasons, "reasons": reasons, "baseline_heldout": b_held["accuracy"], "candidate_heldout": c_held["accuracy"],
            "baseline_golden": b_gold["accuracy"], "candidate_golden": c_gold["accuracy"]}


# ============================================================ candidate training
@dataclass
class TrainResult:
    ran: bool
    reasons: list = field(default_factory=list)
    version: str = ""
    metrics: dict = field(default_factory=dict)
    deployed: bool = False


def training_prerequisites(manifest, cfg):
    """Everything that must hold before weights are touched. Empty list == ready."""
    missing = []
    if manifest["train"] < cfg.get("min_examples", 200):
        missing.append(f"only {manifest['train']} validated training examples (need {cfg.get('min_examples', 200)})")
    for mod in ("torch", "transformers", "peft"):
        if importlib.util.find_spec(mod) is None:
            missing.append(f"python package '{mod}' is not installed (pip install torch transformers peft)")
    if not cfg.get("base_hf"):
        missing.append("LILI_BASE_HF is not set (the Hugging Face id of the SAME base model as your Ollama model)")
    if not cfg.get("ollama_base"):
        missing.append("LILI_MODEL is not set")
    return missing


def train_candidate(memory, registry, data_dir, cfg, ollama_gen_factory=None, runner=None):
    """Optional Level 3. Returns TrainResult. `runner` lets tests inject a fake trainer; the real one needs torch+peft
    and is NOT exercised by the automated tests."""
    manifest = export_training_data(memory, data_dir)
    missing = training_prerequisites(manifest, cfg) if runner is None else (
        [] if manifest["train"] >= cfg.get("min_examples", 200) else [f"only {manifest['train']} validated training examples"])
    if missing:
        return TrainResult(False, missing)
    ver_dir = os.path.join(data_dir, f"candidate_{int(time.time())}")
    try:
        adapter = (runner or _lora_train)(os.path.join(data_dir, "train.jsonl"), ver_dir, cfg)
        name = f"{registry.current().split(':')[0]}-lili-{int(time.time())}"
        if runner is None:
            _ollama_create(name, cfg["ollama_base"], adapter)
    except Exception as e:  # noqa: BLE001 - reported to the user, never hidden
        return TrainResult(False, [f"training failed: {type(e).__name__}: {e}"])
    held = [json.loads(l) for l in open(os.path.join(data_dir, "eval.jsonl"), encoding="utf-8")]
    base_gen, cand_gen = ollama_gen_factory(registry.current()), ollama_gen_factory(name)
    metrics = evaluate_candidate(base_gen, cand_gen, held, min_gain=cfg.get("min_gain", 0.0))
    v = registry.register_candidate(name, metrics)
    if metrics["passed"]:
        registry.deploy(v)
        return TrainResult(True, [], v, metrics, True)
    return TrainResult(True, ["candidate rejected: " + "; ".join(metrics["reasons"])], v, metrics, False)


def _lora_train(train_path, out_dir, cfg):  # pragma: no cover - needs torch/peft/GPU; not run by tests
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_hf"], torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.get("lr", 2e-4))
    rows = [json.loads(l)["messages"] for l in open(train_path, encoding="utf-8")]
    model.train()
    for _ in range(cfg.get("epochs", 2)):
        for msgs in rows:
            prompt = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
            full = prompt + msgs[2]["content"] + (tok.eos_token or "")
            ids = tok(full, return_tensors="pt", truncation=True, max_length=1536).to(dev)
            labels = ids["input_ids"].clone()
            labels[:, :len(tok(prompt)["input_ids"])] = -100          # learn only the assistant's answer
            loss = model(**ids, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir


def _ollama_create(name, base, adapter_dir):  # pragma: no cover - needs ollama
    mf = os.path.join(adapter_dir, "Modelfile")
    with open(mf, "w") as f:
        f.write(f"FROM {base}\nADAPTER {os.path.abspath(adapter_dir)}\n")
    subprocess.run(["ollama", "create", name, "-f", mf], check=True, capture_output=True, text=True, timeout=1800)
