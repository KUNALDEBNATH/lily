"""LILI memory: one SQLite file holding the World Graph and four memory kinds.

Semantic   (facts)        what LILI knows, with source / confidence / status
Episodic   (episodes)     what happened, deduplicated by signature
Procedural (skills)       reusable workflows; only "verified" after an outcome check
Corrective (corrections)  what the user taught her (never auto-expires)
World Graph (nodes/edges) how apps, controls, concepts and actions connect

Rules enforced here (not left to callers):
  * inferred / web facts can never silently become user_confirmed;
  * a user-confirmed fact is never overwritten by a weaker one;
  * secrets (passwords, tokens, card numbers) are refused;
  * "temporary" corrections are refused: they live in the agent session only.
"""
import hashlib
import json
import re
import sqlite3
import time

STATUS_RANK = {"user_confirmed": 0, "web": 1, "inferred": 2, "disputed": 3, "superseded": 9}
CORRECTION_SCOPES = ("action", "preference", "app_fact", "skill")

_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(password|passwd|pwd|passphrase)\b\s*(is|:|=)\s*\S+"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|bearer)\b\s*(is|:|=)?\s*[A-Za-z0-9_\-\.]{12,}"),
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),                       # card-like numbers
    re.compile(r"\b(sk|ghp|gho|xox[abp]|AKIA)[-_A-Za-z0-9]{12,}"),  # common key prefixes
    re.compile(r"(?i)\botp\b\s*(is|:|=)?\s*\d{4,8}"),
]


def contains_secret(text):
    return bool(text) and any(p.search(text) for p in _SECRET_PATTERNS)


def redact(text):
    out = text or ""
    for p in _SECRET_PATTERNS:
        out = p.sub("[REDACTED]", out)
    return out


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def sig(*parts):
    return hashlib.sha1("|".join(norm(str(p)) for p in parts).encode()).hexdigest()[:16]


def tokens(text):
    return set(re.findall(r"[a-z0-9]{2,}", norm(text)))


class MemoryError_(Exception):
    pass


class Memory:
    def __init__(self, path=":memory:"):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL") if path != ":memory:" else None
        self._init()

    # ---------------------------------------------------------------- schema
    def _init(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(id INTEGER PRIMARY KEY, kind TEXT, name TEXT, app TEXT DEFAULT '',
            UNIQUE(kind,name,app));
        CREATE TABLE IF NOT EXISTS edges(src INTEGER, rel TEXT, dst INTEGER, weight REAL DEFAULT 1,
            UNIQUE(src,rel,dst));
        CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, concept TEXT, app TEXT DEFAULT '', meaning TEXT,
            source TEXT, confidence REAL, status TEXT, context TEXT DEFAULT '', retrieved_at TEXT,
            updated_at REAL, UNIQUE(concept,app,meaning));
        CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY, sig TEXT UNIQUE, ts REAL, instruction TEXT,
            steps TEXT, outcome TEXT, lesson TEXT, count INTEGER DEFAULT 1);
        CREATE TABLE IF NOT EXISTS skills(id INTEGER PRIMARY KEY, name TEXT UNIQUE, trigger TEXT, params TEXT,
            steps TEXT, expected TEXT, recovery TEXT, verified_runs INTEGER DEFAULT 0, failures INTEGER DEFAULT 0,
            status TEXT DEFAULT 'candidate', updated_at REAL);
        CREATE TABLE IF NOT EXISTS corrections(id INTEGER PRIMARY KEY, ts REAL, text TEXT, scope TEXT,
            app TEXT DEFAULT '', active INTEGER DEFAULT 1, UNIQUE(text,scope,app));
        CREATE TABLE IF NOT EXISTS failures(id INTEGER PRIMARY KEY, sig TEXT UNIQUE, instruction TEXT, step TEXT,
            context TEXT, reason TEXT, count INTEGER DEFAULT 1, last_ts REAL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    # ----------------------------------------------------------- world graph
    def node(self, kind, name, app=""):
        name, app = norm(name), norm(app)
        self.db.execute("INSERT OR IGNORE INTO nodes(kind,name,app) VALUES(?,?,?)", (kind, name, app))
        row = self.db.execute("SELECT id FROM nodes WHERE kind=? AND name=? AND app=?", (kind, name, app)).fetchone()
        self.db.commit()
        return row["id"]

    def link(self, src, rel, dst):
        """src/dst are (kind, name, app) tuples."""
        a, b = self.node(*src), self.node(*dst)
        self.db.execute("INSERT INTO edges(src,rel,dst,weight) VALUES(?,?,?,1) "
                        "ON CONFLICT(src,rel,dst) DO UPDATE SET weight=weight+1", (a, rel, b))
        self.db.commit()

    def neighbors(self, name, rel=None):
        name = norm(name)
        q = ("SELECT n1.name AS src, e.rel, n2.name AS dst, n2.kind AS dst_kind, n2.app AS app, e.weight "
             "FROM edges e JOIN nodes n1 ON e.src=n1.id JOIN nodes n2 ON e.dst=n2.id WHERE n1.name=?")
        args = [name]
        if rel:
            q += " AND e.rel=?"
            args.append(rel)
        return [dict(r) for r in self.db.execute(q + " ORDER BY e.weight DESC", args)]

    # ------------------------------------------------------------ semantic
    def add_fact(self, concept, meaning, app="", source="inferred", confidence=0.5, context="",
                 user_confirmed=False, retrieved_at=None):
        """Returns the fact id. user_confirmed may ONLY be set by passing the flag explicitly."""
        if contains_secret(concept) or contains_secret(meaning) or contains_secret(context):
            raise MemoryError_("refusing to store text that looks like a secret")
        concept, app, meaning = norm(concept), norm(app), (meaning or "").strip()
        if not concept or not meaning:
            raise MemoryError_("concept and meaning are required")
        status = "user_confirmed" if user_confirmed else ("web" if source == "web" else "inferred")
        if user_confirmed:
            confidence, source = 1.0, "user"
        else:
            confidence = min(float(confidence), 0.9)   # unverified facts are capped below certainty
        existing = self.db.execute("SELECT * FROM facts WHERE concept=? AND app=? AND status!='superseded'",
                                   (concept, app)).fetchall()
        for row in existing:
            if norm(row["meaning"]) == norm(meaning):             # duplicate: strengthen, never downgrade
                new_status = row["status"] if STATUS_RANK[row["status"]] <= STATUS_RANK[status] else status
                self.db.execute("UPDATE facts SET status=?, confidence=max(confidence,?), updated_at=? WHERE id=?",
                                (new_status, confidence, time.time(), row["id"]))
                self.db.commit()
                return row["id"]
        if user_confirmed:                                         # user truth supersedes everything conflicting
            self.db.execute("UPDATE facts SET status='superseded' WHERE concept=? AND app=? AND status!='superseded'",
                            (concept, app))
        elif any(r["status"] == "user_confirmed" for r in existing):
            status = "disputed"                                    # conflicts with user truth: keep but demote
        cur = self.db.execute(
            "INSERT INTO facts(concept,app,meaning,source,confidence,status,context,retrieved_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (concept, app, meaning, source, confidence, status, context, retrieved_at, time.time()))
        self.db.commit()
        self.link(("concept", concept, app), "means", ("meaning", meaning[:80], ""))
        if app:
            self.link(("app", app, ""), "has_control", ("concept", concept, app))
        return cur.lastrowid

    def confirm_fact(self, fact_id):
        """The only path from inferred/web to user_confirmed. Call only after the user said yes."""
        row = self.db.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        if not row:
            raise MemoryError_("no such fact")
        self.db.execute("UPDATE facts SET status='superseded' WHERE concept=? AND app=? AND id!=? "
                        "AND status!='superseded'", (row["concept"], row["app"], fact_id))
        self.db.execute("UPDATE facts SET status='user_confirmed', source='user', confidence=1.0, updated_at=? "
                        "WHERE id=?", (time.time(), fact_id))
        self.db.commit()

    def get_facts(self, concept, app="", cross_app=True):
        """Best-first. Falls back to other apps (flagged) when nothing is known in this app."""
        concept, app = norm(concept), norm(app)
        rows = self.db.execute("SELECT * FROM facts WHERE concept=? AND app=? AND status!='superseded'",
                               (concept, app)).fetchall()
        out = [dict(r, cross_app=False) for r in rows]
        if not out and cross_app:
            rows = self.db.execute("SELECT * FROM facts WHERE concept=? AND status!='superseded'", (concept,)).fetchall()
            out = [dict(r, cross_app=True) for r in rows]
        return sorted(out, key=lambda r: (STATUS_RANK.get(r["status"], 5), -r["confidence"]))

    def knows(self, concept, app=""):
        return bool(self.get_facts(concept, app))

    def search_facts(self, text, app="", limit=5):
        want = tokens(text)
        scored = []
        for r in self.db.execute("SELECT * FROM facts WHERE status!='superseded'"):
            overlap = len(want & (tokens(r["concept"]) | tokens(r["meaning"])))
            if overlap and (not app or r["app"] in ("", norm(app))):
                scored.append((overlap, -STATUS_RANK.get(r["status"], 5), dict(r)))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [r for _, _, r in scored[:limit]]

    def delete_fact(self, concept, app=None):
        q, a = "DELETE FROM facts WHERE concept=?", [norm(concept)]
        if app is not None:
            q += " AND app=?"
            a.append(norm(app))
        n = self.db.execute(q, a).rowcount
        self.db.commit()
        return n

    # ------------------------------------------------------------ episodic
    def add_episode(self, instruction, steps, outcome, lesson=""):
        instruction, lesson = redact(instruction), redact(lesson)
        s = sig(instruction, outcome, json.dumps(steps, sort_keys=True))
        row = self.db.execute("SELECT id FROM episodes WHERE sig=?", (s,)).fetchone()
        if row:
            self.db.execute("UPDATE episodes SET count=count+1, ts=? WHERE id=?", (time.time(), row["id"]))
        else:
            self.db.execute("INSERT INTO episodes(sig,ts,instruction,steps,outcome,lesson) VALUES(?,?,?,?,?,?)",
                            (s, time.time(), instruction, json.dumps(steps), outcome, lesson))
        self.db.commit()
        return s

    def episodes(self, outcome=None, limit=100):
        q, a = "SELECT * FROM episodes", []
        if outcome:
            q += " WHERE outcome=?"
            a.append(outcome)
        return [dict(r, steps=json.loads(r["steps"])) for r in self.db.execute(q + " ORDER BY ts DESC LIMIT ?", a + [limit])]

    def prune_episodes(self, keep=2000):
        self.db.execute("DELETE FROM episodes WHERE id NOT IN (SELECT id FROM episodes ORDER BY ts DESC LIMIT ?)", (keep,))
        self.db.commit()

    # ---------------------------------------------------------- procedural
    def save_skill(self, name, trigger, steps, expected, params=None, recovery=None):
        """New skills start as 'candidate'. Only record_skill_run(success=True, verified=True) promotes."""
        if contains_secret(json.dumps(steps)):
            raise MemoryError_("refusing to store a skill containing a secret")
        old = self.db.execute("SELECT id FROM skills WHERE name=?", (name,)).fetchone()
        vals = (trigger, json.dumps(params or {}), json.dumps(steps), json.dumps(expected), json.dumps(recovery or []),
                time.time(), name)
        if old:
            self.db.execute("UPDATE skills SET trigger=?, params=?, steps=?, expected=?, recovery=?, updated_at=? "
                            "WHERE name=?", vals)
        else:
            self.db.execute("INSERT INTO skills(trigger,params,steps,expected,recovery,updated_at,name) "
                            "VALUES(?,?,?,?,?,?,?)", vals)
        self.db.commit()

    def record_skill_run(self, name, success, verified):
        row = self.get_skill(name)
        if not row:
            return None
        if success and verified:
            self.db.execute("UPDATE skills SET verified_runs=verified_runs+1, status='verified', failures=0 WHERE name=?",
                            (name,))
        elif not success:
            self.db.execute("UPDATE skills SET failures=failures+1 WHERE name=?", (name,))
            if row["failures"] + 1 >= 3 and row["failures"] + 1 > row["verified_runs"]:
                self.db.execute("UPDATE skills SET status='retired' WHERE name=?", (name,))
            elif row["status"] == "verified":
                self.db.execute("UPDATE skills SET status='candidate' WHERE name=?", (name,))  # needs re-verification
        self.db.commit()
        return self.get_skill(name)

    def demote_skill(self, name):
        """Used when the user says a skill's method was wrong: it must be re-verified before automatic reuse."""
        self.db.execute("UPDATE skills SET status='candidate' WHERE name=? AND status='verified'", (name,))
        self.db.commit()

    def get_skill(self, name):
        r = self.db.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        return self._skill(r) if r else None

    @staticmethod
    def _skill(r):
        d = dict(r)
        for k in ("params", "steps", "expected", "recovery"):
            d[k] = json.loads(d[k])
        return d

    def find_skill(self, instruction, verified_only=True):
        want = tokens(instruction)
        best, best_score = None, 0.0
        for r in self.db.execute("SELECT * FROM skills WHERE status!='retired'"):
            if verified_only and r["status"] != "verified":
                continue
            trig = tokens(re.sub(r"\{[a-z_]+\}", " ", r["trigger"]))
            if not trig:
                continue
            score = len(want & trig) / len(trig)
            if score >= 0.6 and score > best_score:
                best, best_score = r, score
        return self._skill(best) if best else None

    def delete_skill(self, name):
        n = self.db.execute("DELETE FROM skills WHERE name=?", (name,)).rowcount
        self.db.commit()
        return n

    # ---------------------------------------------------------- corrective
    def add_correction(self, text, scope, app=""):
        if scope == "temporary":
            raise MemoryError_("temporary instructions must not be persisted")
        if scope not in CORRECTION_SCOPES:
            raise MemoryError_(f"unknown scope {scope!r}")
        if contains_secret(text):
            raise MemoryError_("refusing to store text that looks like a secret")
        self.db.execute("INSERT INTO corrections(ts,text,scope,app) VALUES(?,?,?,?) "
                        "ON CONFLICT(text,scope,app) DO UPDATE SET active=1, ts=excluded.ts",
                        (time.time(), text.strip(), scope, norm(app)))
        self.db.commit()

    def corrections_for(self, instruction="", app=""):
        """Preferences always apply; action/skill/app_fact corrections apply when they overlap the task."""
        want, app = tokens(instruction), norm(app)
        out = []
        for r in self.db.execute("SELECT * FROM corrections WHERE active=1 ORDER BY ts DESC"):
            if r["app"] and app and r["app"] != app:
                continue
            if r["scope"] == "preference" or len(want & tokens(r["text"])) >= 2 or (r["app"] and r["app"] == app):
                out.append(dict(r))
        return out

    def remove_correction(self, text):
        n = self.db.execute("DELETE FROM corrections WHERE text=?", (text.strip(),)).rowcount
        self.db.commit()
        return n

    # ------------------------------------------------------------ failures
    def record_failure(self, instruction, step, context, reason):
        s = sig(step, context)
        row = self.db.execute("SELECT id FROM failures WHERE sig=?", (s,)).fetchone()
        if row:
            self.db.execute("UPDATE failures SET count=count+1, reason=?, last_ts=? WHERE id=?",
                            (reason, time.time(), row["id"]))
        else:
            self.db.execute("INSERT INTO failures(sig,instruction,step,context,reason,last_ts) VALUES(?,?,?,?,?,?)",
                            (s, redact(instruction), step, context, reason, time.time()))
        self.db.commit()
        return s

    def failure_count(self, step, context):
        """Failures are keyed on step+context, so a changed context (new page/app state) is a fresh attempt."""
        row = self.db.execute("SELECT count FROM failures WHERE sig=?", (sig(step, context),)).fetchone()
        return row["count"] if row else 0

    def failures(self, limit=50):
        return [dict(r) for r in self.db.execute("SELECT * FROM failures ORDER BY last_ts DESC LIMIT ?", (limit,))]

    # --------------------------------------------------------------- stats
    def stats(self):
        t = lambda name: self.db.execute(f"SELECT COUNT(*) c FROM {name}").fetchone()["c"]
        return {k: t(k) for k in ("nodes", "edges", "facts", "episodes", "skills", "corrections", "failures")}
