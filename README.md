<<<<<<< HEAD
# LILI

A lightweight, voice-to-voice computer-use agent: she listens, speaks (female SAPI voice), operates a browser and
Windows apps through semantic interfaces (not screenshots), asks about things she doesn't understand, remembers what you
teach her, and only learns a skill after its outcome was verified.

**Status, honestly.** Everything below was developed and tested in a Linux sandbox. The automated suite passes there
(115 passed, 4 live tests skipped). **Not yet run on a real Windows machine:** SAPI voice, microphone capture,
faster-whisper, Windows UI Automation, pywinauto desktop control, a real Ollama model. Section 19 lists what you must check
locally. Nothing here has been benchmarked against Nova, UFO², or on OSWorld.

## 1. Purpose
Say "LILI, open YouTube and search for …" and she does it, verifies it worked, and tells you. If she meets an unfamiliar
consequential control ("Archive"), she asks you once, stores your answer as user-confirmed knowledge, and uses it later.

## 2. Architecture (as implemented)
```
mic -> VAD -> faster-whisper -> wake-word/echo filter -> Agent.handle()
   routing: stop | answer-to-my-question | correction | investigate | task
   task: curiosity check (memory first, ask only if it matters)
         -> 1) verified skill (0 model calls)  2) built-in plan (0 calls)  3) local LLM, one schema-validated decision/step
         -> SafetyGate -> Executor (preconditions, timeout, limits, cancel) -> Verifier (explicit expected outcome)
         -> success: episode + skill (only if verified)   failure: recorded per (step, context), replan
   perception: Browser DOM (MutationObserver cache) > Windows UIA > OCR > OpenCV template match
   memory: one SQLite file (facts, episodes, skills, corrections, failures, graph nodes/edges)
   level 3 (optional): export -> validate -> dedupe -> split -> LoRA -> regression gate -> versioned deploy -> rollback
reply -> SAPI TTS (killed instantly if you talk over her)
```
No optional visual-language model is implemented (priority level 5 in the brief). Unlabeled icons in custom-drawn UIs
that UIA/OCR cannot name can only be matched if you supply a template; otherwise LILI will ask you or say she cannot see it.

## 3. Files (8 Python files, the maximum)
| file | contents |
|---|---|
| `main.py` | CLI, voice and text sessions, `doctor`, simulated world, `self-test`, `benchmark` |
| `agent.py` | routing, curiosity hookup, skill/plan/LLM loop, recovery, replies |
| `model.py` | Ollama adapter, schema validator, VAD, microphone, faster-whisper, SAPI TTS, barge-in, echo filter |
| `perception.py` | DOM, UIA, OCR, OpenCV, cache, element resolution |
| `actions.py` | tool schemas, SafetyGate, prompt-injection sanitiser, Executor, Verifier |
| `memory.py` | SQLite World Graph + semantic / episodic / procedural / corrective memory |
| `learning.py` | curiosity, correction classifier, skill learning, training export, evaluation gate, registry, LoRA hook |
| `test_agent.py` | the test suite |

Plus `README.md`, `requirements.txt`, `.env.example`. Verify the limit: `(Get-ChildItem *.py).Count` must print 8 or less.

## 4. Research references
Verified this session (page content actually read):
- UFO² Desktop AgentOS (Microsoft Research): HostAgent/AppAgent split, hybrid UIA + vision control detection, speculative multi-action planning. https://www.microsoft.com/en-us/research/publication/ufo2-the-desktop-agentos/
- OSWorld 2.0 write-up (XLANG / Snorkel): top agent fully completes 20.6% of tasks; failures dominated by missing information, perception, *premature "done" without verification*, and memory over long runs. https://snorkel.ai/blog/osworld-2-0-why-computer-use-agents-fail-most-tasks/
- faster-whisper on PyPI: MIT licence, CPU `int8`, models auto-download on first use, FFmpeg not required; release 1.2.1 (31 Oct 2025); classifiers list Python 3.9-3.11. https://pypi.org/project/faster-whisper/
- Ollama structured outputs: JSON schema in the `format` field of `/api/chat`, temperature 0 recommended. https://registry.ollama.ai/blog/structured-outputs

**Not verified** (the pages were blocked in this session, or no usable result was found; listed from prior knowledge only): Nova AI
(I found nothing I could cite, so I make **no** claim about it), Voyager (skill library), Agent Workflow Memory, Silero VAD,
Piper TTS, pywinauto licence/maintenance, sounddevice, the Ollama `qwen3:4b` tag and its tool-calling quality, Playwright.
Check these yourself before relying on them. The two research rounds were therefore narrower than the brief asked for.

## 5. Technical choices and why (including what the "challenge" round changed)
- **No model call per step.** Verified skills and built-in plans run with zero model calls (measured in simulation: 3 -> 0).
- **No screenshots by default.** DOM extraction is cached and re-read only when a MutationObserver or URL change says the page changed.
- **VAD:** a ~60-line numpy energy detector with adaptive noise floor instead of Silero VAD, to avoid pulling in torch. Trade-off:
  weaker in noisy rooms. Swap it if it mis-triggers.
- **TTS:** Windows SAPI via a PowerShell child process: offline, zero download, trivially interruptible (kill the process). Trade-off:
  less natural than neural voices such as Piper (not implemented). Text goes over stdin, never in the command line.
- **STT:** faster-whisper `base.en`, int8 on CPU. Larger models are more accurate and slower; unmeasured here.
- **Verified-only learning:** OSWorld 2.0's finding that agents declare success without checking motivated this: no expected outcome means "not verified", and unverified work never becomes a skill.
- **Failures are keyed on (step, page context)**, so a changed page is a fresh attempt rather than a permanent ban.
- **Inferred knowledge never silently becomes confirmed**; for consequential controls an inferred guess is *asked*, not trusted.
- Memory-vs-weights are separate: facts are learned instantly; weights change only through the gated Level 3 pipeline.
- Not novel: skill libraries, graph memory and human feedback exist in prior work. The only claim here is this particular combination in one small codebase, and that claim is **untested** against alternatives.

## 6-7. Voice and the female voice
Voice is SAPI. LILI picks `LILI_VOICE` if set, else the first installed Female voice. She never hard-codes a voice: if none exists,
`doctor` and `voice-test` print what is installed and how to add one. Rate: `LILI_RATE` (-10..10), volume: `LILI_VOLUME`.
Wake phrase "LILI" is detected on the **local transcript** (every speech segment is transcribed locally and discarded unless addressed to her).
After you interrupt her, the next utterance is treated as addressed to her for 15 s. Raw audio is never written to disk.
Her own speech is filtered out by comparing transcripts to what she just said. Headphones make barge-in far more reliable.

## 8. Local model
Default `qwen3:4b` through Ollama: an initial candidate **I have not benchmarked**. Try a smaller/larger one with `$env:LILI_MODEL`.
Without a model LILI still does the built-in plans (open a site, search YouTube) and says so. Nothing is downloaded automatically.

## 9-15. Setup, in order (Windows PowerShell)
```powershell
# 1. Python (3.10+; faster-whisper's PyPI page lists 3.9-3.11, so 3.11 is the safe choice)
python --version

# 2. Dependencies
pip install -r requirements.txt

# 3. Browser
python -m playwright install chromium

# 4. Local model (installer from https://ollama.com ; or: winget install Ollama.Ollama)
ollama pull qwen3:4b

# 5. Optional OCR: install the Tesseract program (UB-Mannheim build) and add it to PATH

# 6. Voices: list installed voices; add a female one in Settings > Time & language > Speech > Manage voices
Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | % { $_.VoiceInfo.Name + " | " + $_.VoiceInfo.Gender }
$env:LILI_VOICE = "Microsoft Zira Desktop"   # only if you want a specific one (use a name printed above)

# 7. Diagnostics (read every [FAIL] / [warn] line and its fix)
python main.py doctor

# 8. Tests and self-test (safe: simulated, no real mail/files/purchases/desktop)
python -m pytest -q
python main.py self-test

# 9. Check microphone, speech recognition and voice (first run downloads the speech model once)
python main.py voice-test

# 10. Start
python main.py run --mode voice
python main.py run --mode text      # typed fallback
```
Other commands: `python main.py benchmark` (deterministic *simulated* measurements, prints what it did NOT measure),
`python main.py export-training-data`, `python main.py train-candidate`.
Run the live opt-in tests on your machine: `$env:LILI_LIVE = "1"; python -m pytest -q -k Live`.
Sign in to sites yourself in the browser window LILI opens (profile saved in `LILI_HOME`); she never types passwords.

## 16. Memory and learning
One SQLite file `LILI_HOME\lili.db`. Facts carry source, confidence, status (`user_confirmed` > `web` > `inferred` > `disputed`) and context.
Your answer to a question becomes `user_confirmed` and supersedes conflicting guesses. Corrections are classified as
action / preference / app-fact / skill / **temporary** (temporary applies to the next task only and is never stored).
Passwords, keys and card-like numbers are refused. Skills start as `candidate`, become `verified` only after a task whose expected
outcome was confirmed, drop back when you correct them, and are retired after repeated failures. Duplicate experiences are merged by signature.
Say "remember what I taught you" for a summary.

## 17. Optional model training (Level 3)
`export-training-data` writes validated, de-duplicated, train/eval-split examples (secrets excluded, failures excluded, your rules included).
`train-candidate` refuses to run until: >= `LILI_MIN_TRAIN` validated examples, `torch transformers peft` installed, `LILI_BASE_HF` set to the same base model
as your Ollama model. Then it LoRA-trains, builds an Ollama model, evaluates it against the baseline on held-out data plus golden regression cases, and deploys it only
if nothing regressed; the previous model is always kept (`ModelRegistry.rollback`). **The real LoRA step has never been run** (no GPU/torch here). The tests
check the gating, deploy and rollback logic with an injected stand-in trainer; they say nothing about training quality. Do not assume it works until you've run it and read the report.

## 18. Safety controls
Model output is parsed as JSON and checked against strict per-tool schemas; unknown tools and extra fields are rejected; no code is ever executed.
Confirmation (spoken "yes") is required for send/delete/buy/pay/submit/post-style controls, Enter on forms, form submits, card-like text, and anything after suspicious page text.
Never types into password fields. Only http(s) URLs. Blocked: Alt+F4, Win+R, Ctrl+Alt+Del, Task Manager shortcuts; apps limited to an allow-list (notepad, calc, paint, explorer).
Page text is untrusted: instruction-like sentences are stripped before the model sees them, and a hijacked model still can't pass the gate. Limits: 25 actions and 12 model calls per task.
Stop: say "LILI, stop", press Ctrl+C, or slam the mouse into a screen corner (pyautogui fail-safe). Say "goodbye LILI" to quit.
The sanitiser is a pattern list, not a guarantee; treat the gate, not the filter, as the real defence.

## 19. Limitations / still to verify on your PC
- Untested on real Windows: SAPI voice + barge-in, mic/VAD thresholds, faster-whisper, UIA reading, pywinauto clicks, Ollama + qwen3:4b decision quality and latency.
- Latency is printed per voice turn (`[latency] ...`); I have **no real numbers**. Only simulated action latencies (ms) exist.
- No OSWorld run, no comparison with Nova/UFO² (conditions not comparable or not available).
- Windows desktop *actions* are limited (click/type via UIA, hotkeys, open allow-listed apps); desktop *perception* is foreground-window only.
- Energy VAD can false-trigger on noise; wake phrase relies on transcription of "lili" (may be heard as "lily").
- "Compare / best / most relevant" judgement depends entirely on the small model's quality.

## 20. Troubleshooting
| symptom | fix |
|---|---|
| `doctor`: microphone FAIL | install `sounddevice`; enable the mic in Windows privacy settings; set the default input device |
| "no female voice installed" | add a voice (step 6) or set `LILI_VOICE` to an installed name |
| "model not installed" / "Ollama not reachable" | start Ollama, `ollama pull qwen3:4b` |
| first voice run hangs | faster-whisper is downloading its model once; or set `LILI_STT_DIR` to a local copy |
| she keeps hearing herself | use headphones; raise nothing else first |
| she ignores you | start with "LILI, ..."; check `heard:` lines in the console |
| browser won't start | `python -m playwright install chromium`, or `$env:LILI_BROWSER_CHANNEL = "msedge"` |
| `pip install faster-whisper` fails on Python 3.13 | install Python 3.11 (PyPI lists 3.9-3.11) |
=======
# lily
>>>>>>> 94c2d8ac78cc95c7214a7b798da51793f9b08a2c
