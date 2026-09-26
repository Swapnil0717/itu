#!/usr/bin/env python3
"""
ITU-1 -- Step 4, free/local variant of claude_label.py.

Identical pipeline to claude_label.py (same SYSTEM_PROMPT, same grounding
verification, same architecture.assemble() call, same TrainingExample
shape) -- the only thing that changes is where the labeling engine's text
comes from: instead of POSTing to api.anthropic.com with an
ANTHROPIC_API_KEY, this calls a LOCAL Ollama server. No API key, no
per-request cost, nothing leaves your machine.

Setup (one-time)
-----------------
1. Install Ollama: https://ollama.com/download  (Windows/Mac/Linux, free)
2. Pull a model that's reasonably good at following JSON instructions.
   Bigger = better instruction-following = more issues actually get
   a value instead of silently degrading to UNKNOWN. If your machine can
   run it, prefer something in the 8B+ instruct-tuned range, e.g.:

       ollama pull llama3.1:8b
       ollama pull qwen2.5:7b-instruct

   (qwen2.5 tends to be noticeably more reliable at strict JSON output).
3. Make sure the Ollama server is running (it starts automatically after
   install / on login; `ollama list` should work in a terminal without
   errors).

Usage
-----
    python local_label.py \
        --collected data/collected.jsonl \
        --weak-labels data/weak_labels.jsonl \
        --batches data/labeling_batches.jsonl \
        --out data/training_examples.jsonl \
        --spot-check-out data/spot_check_needed.jsonl \
        --rejects-out data/rejected.jsonl \
        --spot-check-rate 0.175 \
        --model llama3.1:8b \
        --limit 10          # small test batch first, same as claude_label.py

Then drop --limit to run the rest. --apply-corrections works exactly like
claude_label.py (re-derives/re-validates from data/corrections.jsonl,
doesn't call the model again).

Reality check on quality
-------------------------
A local 7-8B model will be noticeably weaker than Claude at: staying
strictly inside the JSON contract, correctly choosing EXPLICIT vs
INFERRED vs UNKNOWN, and not hallucinating evidence_text. This script
compensates where it safely can (format="json" constraint, a
regex-fallback JSON extractor, retries on unparseable output) but the
verify_grounding() step is your real safety net -- it already discards
any claim whose evidence_text doesn't verbatim-match the source segment,
same as claude_label.py. Keep the --spot-check-rate meaningfully high
(the 0.175 default is a reasonable floor) since a weaker model means the
sampled error rate you find is likely higher than Claude would have given
you, and that's the number that (per the original build prompt) estimates
the error rate across the whole unreviewed set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "itu1"))
import architecture  # noqa: E402
import example as example_mod  # noqa: E402
from derived import finalize, validate_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_issues import is_labelable  # noqa: E402

TOOL_VERSION = "local-label-0.1"

CLASSIFICATION_FIELDS = ("role", "experience_level", "complexity", "task_type")
LIST_FIELDS = ("technologies", "languages", "frameworks", "technical_areas",
                "components", "systems", "affected_areas")

# Same contract claude_label.py uses -- keep in sync if you edit either.
SYSTEM_PROMPT = """You are labeling a GitHub issue for the ITU-1 dataset. \
You are given numbered TEXT SEGMENTS (verbatim, never paraphrase them) and \
must output a JSON array of field proposals -- nothing else, no markdown \
fences, no commentary.

Each element of the array must be an object:
{
  "field": "<field name>",
  "value": <the value -- string, list of strings, or the scope object>,
  "source": "EXPLICIT" | "SUPPORTED_BY_CONTEXT" | "INFERRED" | "UNKNOWN",
  "pointers": ["S1", "S3", ...],   // segment_ids this claim is grounded in; [] if UNKNOWN
  "evidence_text": "<verbatim substring copied from a pointed segment>" | null,
  "confidence": <0.0-1.0> | null   // null only when source is UNKNOWN
}

Grounding rules (violating these gets the claim silently discarded):
- evidence_text must be an exact, verbatim substring of the text of a
  segment you listed in pointers. Never paraphrase, summarize, or
  reconstruct a quote from memory -- copy it exactly.
- source=EXPLICIT only when the issue's own title/body/label text directly
  states the value. source=SUPPORTED_BY_CONTEXT when a comment or label
  supports it. source=INFERRED when you are inferring from indirect signal
  (e.g. judging complexity from described scope). source=UNKNOWN with no
  pointers/evidence/confidence when you have no real signal -- prefer
  UNKNOWN over a low-confidence guess.

Required fields, every time (use source=UNKNOWN + empty/appropriate-default
value when you have no signal -- do not omit the field):
  task_type   one of: Bug, Feature, Improvement, Refactor, Performance,
              Security, Maintenance, Documentation, Other, Unknown
  role        one of: Frontend, Backend, Fullstack, Unknown
  experience_level  one of: Beginner, Intermediate, Advanced, Unknown
  complexity  one of: Low, Medium, High, Unknown
  title       short string
  summary     1-3 sentence string
  objective   string: what needs to be done
  expected_outcome  string: what "done" looks like
  acceptance_criteria  list of strings (can be [] with source UNKNOWN)
  scope       {"in_scope": [...], "out_of_scope": [...]} (empty lists OK)

Optional fields, include only when you have real signal, otherwise omit
entirely (do not send empty-UNKNOWN versions of these -- the caller fills
defaults for any you skip):
  technologies, languages, frameworks, technical_areas, components,
  systems, affected_areas (all lists of strings), dependencies (list of
  {"type": "blocks"|"blocked_by"|"relates_to"|"requires", "ref": "...",
  "description": "..."})

EXAMPLE -- follow this shape EXACTLY. Given:
  [S1] (ISSUE_TITLE) Add dark mode toggle to settings page
  [S2] (ISSUE_BODY) Users want a dark mode option, a simple UI toggle in
  Settings that switches the app theme. No backend changes needed.

Correct output (every field gets its OWN object like this -- never merge
fields into one flat {"task_type": "...", "role": "..."} object):
[
  {"field": "task_type", "value": "Feature", "source": "EXPLICIT",
   "pointers": ["S1"], "evidence_text": "Add dark mode toggle to settings page",
   "confidence": 0.9},
  {"field": "role", "value": "Frontend", "source": "EXPLICIT",
   "pointers": ["S2"], "evidence_text": "a simple UI toggle in Settings that switches the app theme",
   "confidence": 0.85},
  {"field": "experience_level", "value": "Unknown", "source": "UNKNOWN",
   "pointers": [], "evidence_text": null, "confidence": null},
  {"field": "complexity", "value": "Low", "source": "INFERRED",
   "pointers": ["S2"], "evidence_text": "No backend changes needed",
   "confidence": 0.6}
  ... (title, summary, objective, expected_outcome, acceptance_criteria,
  scope follow the same one-object-per-field shape; include every
  required field this way, every time)
]

A pre-fill hint from a cheap heuristic may be given below. Treat it as a
suggestion only -- verify against the actual segment text and override or
discard it if it's wrong. Output the JSON array now, in the exact shape
shown in the example above -- never as a flat {field: value} object."""

# Appended to the user prompt on a retry after a flat/malformed first
# response -- names the mistake and repeats the required shape inline so
# the model doesn't need to re-read the full system prompt to self-correct.
_CORRECTIVE_NUDGE = """

Your previous answer did not follow the required format -- it looked like \
a flat {"task_type": "...", "role": "...", ...} object instead of a JSON \
array of per-field objects. This is a hard requirement, not a style \
preference. Redo it now as an array where EVERY field is its own object: \
{"field": "...", "value": ..., "source": "EXPLICIT"|"SUPPORTED_BY_CONTEXT"|"INFERRED"|"UNKNOWN", \
"pointers": [...], "evidence_text": "..."|null, "confidence": <0-1>|null}. \
Output ONLY the JSON array, nothing else."""


# ------------------------------------------------------- Ollama API client

def call_ollama(system: str, user: str, model: str, base_url: str,
                 max_retries: int = 3, timeout: int = 180, num_predict: int = 900) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    for attempt in range(max_retries):
        started = time.time()
        try:
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "format": "json",   # constrains output to syntactically valid JSON
                    # num_predict caps how many tokens the model can generate --
                    # without this a slow/rambling model can run far past what
                    # this task needs, which is the main thing that turns into
                    # a timeout on weak CPUs. 900 is generous for one issue's
                    # JSON array but bounds the worst case.
                    "options": {"temperature": 0, "num_predict": num_predict},
                },
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout:
            elapsed = time.time() - started
            print(f"  timed out after {elapsed:.0f}s (limit {timeout}s) on attempt {attempt + 1}/{max_retries}",
                  file=sys.stderr)
            if attempt == max_retries - 1:
                raise
            continue
        except requests.ConnectionError:
            sys.exit(
                f"Could not reach Ollama at {base_url}. Is it running? "
                "Try `ollama list` in a terminal, or install from https://ollama.com/download"
            )
        if resp.status_code == 404 and attempt == 0:
            sys.exit(
                f"Ollama returned 404 for model '{model}'. Did you `ollama pull {model}` first?"
            )
        if resp.status_code >= 500:
            wait = min(30, 2 ** attempt)
            print(f"  Ollama {resp.status_code}, retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    raise RuntimeError("Ollama: exhausted retries")


# ---------------------------------------------------------- prompt building

def render_segments(segments: list["architecture.Segment"]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{s.segment_id}] ({s.type}) {s.text}")
    return "\n".join(lines)


def render_hint(weak: dict | None) -> str:
    if not weak:
        return "(no pre-fill hint available)"
    parts = []
    for field, val in weak.items():
        if val:
            parts.append(f"{field}: {val['value']} (heuristic, confidence {val['confidence']})")
    return "\n".join(parts) if parts else "(no pre-fill hint available)"


# --------------------------------------------------- response parsing

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> str:
    """Local models sometimes wrap the array in prose despite format='json'
    (e.g. {"proposals": [...]}  or a stray leading sentence). Try the plain
    parse first; if that fails, grab the outermost [...] span and, failing
    that, look for a top-level object with one list-valued key."""
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


# Top-level field names the SYSTEM_PROMPT asks for. Used only to recognize
# the flat-map failure mode below -- not a schema check.
_KNOWN_FIELDS = {
    "task_type", "role", "experience_level", "complexity", "title", "summary",
    "objective", "expected_outcome", "acceptance_criteria", "scope",
    "technologies", "languages", "frameworks", "technical_areas", "components",
    "systems", "affected_areas", "dependencies",
}


def _flat_dict_to_proposals(data: dict, anchor_pointers: list[str]) -> list[dict]:
    """Recover a flat {field: value} response into proposal-shaped dicts.

    A known small-model failure mode: the model answers the labeling
    question correctly but can't hold the nested per-field envelope
    (source/pointers/evidence_text/confidence) in its head at the same
    time, so it emits plain {"task_type": "Feature", ...} instead of the
    array of proposal objects the grounding pipeline expects. Rather than
    discard genuinely useful values, recover them here -- but honestly:
    source is forced to INFERRED and confidence is fixed low.

    Two things every INFERRED claim needs downstream, and why they're
    filled the way they are instead of left empty:
    - a *resolvable* pointer, or architecture.assemble()'s H2.6 check
      downgrades the claim straight to UNKNOWN and the value is lost
      again anyway. Each proposal is pointed at the issue's own
      title/body segments -- the same "whole issue, no exact quote"
      grounding the heuristic engine already uses for its own INFERRED
      complexity/experience_level guesses in architecture.py.
    - a non-empty evidence_text string, or schema rule 3 (INFERRED
      requires evidence) rejects the record. Since there's no real quote
      to give, evidence_text is a fixed, honest disclosure string rather
      than an invented one -- verify_grounding()'s verbatim-substring
      check will (correctly) find it doesn't match the segment text and
      leaves the claim as INFERRED with confidence clamped, which is
      exactly the outcome wanted here. The note flags it for spot-check.

    A response after the corrective retry (see propose()) can come back
    HALF-converted -- some fields still bare values, others already
    wrapped as {"value": ..., "source": ..., ...} envelopes because the
    model partially followed the nudge. itu1/schema.py's own enum check
    does `value not in allowed_set`, which raises TypeError (not a clean
    validation error) if value is an unhashable dict -- so any such
    envelope-shaped value is unwrapped to its inner "value" here rather
    than passed through raw. Anything still not a plain string/list/dict
    -scope-shape after that is dropped rather than risk the same crash
    downstream -- losing one field's guess is fine, crashing the whole
    batch on issue N is not."""
    disclosure = "unverified: recovered from a flat (non-grounded) model response, no per-field evidence given"
    proposals = []
    for k, v in data.items():
        if k not in _KNOWN_FIELDS or v is None:
            continue
        if isinstance(v, dict) and "value" in v and any(
                key in v for key in ("source", "confidence", "evidence_text", "pointers")):
            # Half-converted: the model wrapped THIS field correctly on
            # its own -- unwrap rather than nesting the envelope as if it
            # were the raw value.
            v = v["value"]
        if k in CLASSIFICATION_FIELDS and not isinstance(v, str):
            # A classification field (task_type/role/experience_level/
            # complexity) MUST be a plain string for schema.py's enum
            # check to work at all -- anything else (list, dict, number)
            # would crash _check_task_structure with an unhashable-type
            # TypeError rather than a normal validation error. Drop it;
            # the caller's default_fill_proposals() will fill Unknown.
            continue
        if k == "scope" and not isinstance(v, dict):
            continue
        if k in LIST_FIELDS or k == "acceptance_criteria":
            # schema.py requires a list of strings here (_is_str_list) --
            # and for acceptance_criteria specifically, an empty list also
            # trips Rule 5 ("acceptance_criteria is empty but classification
            # succeeded"), so a dropped value fails just as hard as a bad
            # one. A bare string is the model's most common mistake for
            # these fields; coerce it to a one-item list instead of losing
            # it. Anything else malformed (dict, number, list containing
            # non-strings) still gets dropped -- Unknown/[] downstream is
            # safer than guessing at a shape.
            if isinstance(v, str):
                v = [v]
            elif not (isinstance(v, list) and all(isinstance(i, str) for i in v)):
                continue
        proposals.append({
            "field": k, "value": v, "source": "INFERRED",
            "pointers": list(anchor_pointers), "evidence_text": disclosure, "confidence": 0.3,
            "note": "recovered from flat (non-grounded) model response",
        })
    return proposals


def _parse_raw_json(text: str) -> dict | list:
    """Parse-only, no flat-recovery and no shape decisions -- lets the
    caller inspect what the model actually returned (e.g. to decide
    whether a retry is worth it) before any lossy recovery happens.
    Raises ValueError/json.JSONDecodeError on unparseable output."""
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_extract_json_array(text))


def _is_flat_response(data) -> bool:
    return isinstance(data, dict) and any(k in _KNOWN_FIELDS for k in data.keys())


def _is_single_proposal(data) -> bool:
    return isinstance(data, dict) and "field" in data and "value" in data


REQUIRED_FIELD_NAMES = {
    "task_type", "role", "experience_level", "complexity", "title",
    "summary", "objective", "expected_outcome", "acceptance_criteria", "scope",
}
# SYSTEM_PROMPT asks for these 10 fields "every time" (UNKNOWN + a default
# value when there's no signal, but never omitted). A response that covers
# fewer than this many has the model giving up partway through -- whether
# that shows up as a flat dict, unparseable text, or (the case that slipped
# through silently before this check existed) one lone well-formed proposal
# object with no array wrapper around it. All three are worth one corrective
# retry, same rationale the flat/unparseable cases already used.
MIN_REQUIRED_FIELDS_TO_ACCEPT = 5


def _covered_required_fields(data) -> set:
    """Cheap, best-effort count of which of the 10 always-required fields a
    freshly-parsed (not yet normalized) response addresses -- used only to
    decide whether a retry is worth it. parse_claude_response() below is
    still the sole authority on the final, normalized proposal list."""
    if isinstance(data, dict):
        if _is_single_proposal(data):
            field = data.get("field")
            return {field} if field in REQUIRED_FIELD_NAMES else set()
        return set(data.keys()) & REQUIRED_FIELD_NAMES
    if isinstance(data, list):
        return {item.get("field") for item in data if isinstance(item, dict)} & REQUIRED_FIELD_NAMES
    return set()


def parse_claude_response(text: str, anchor_pointers: list[str] | None = None) -> list[dict]:
    """Raises ValueError on unparseable output -- caller treats as a reject.

    anchor_pointers: segment_ids to attach to any flat-recovered proposal
    (see _flat_dict_to_proposals); pass the issue's title/body segment ids.
    This is the LAST-RESORT path (after OllamaEngine.propose's retry has
    already failed to get a properly-shaped response) -- it still recovers
    a flat response rather than losing it, same as before.
    """
    data = _parse_raw_json(text)
    if isinstance(data, dict):
        if _is_flat_response(data):
            # {"task_type": "Feature", "role": "Backend", ...} -- the flat
            # failure mode described above. Checked before the wrapper case
            # below because a flat response can itself contain list-valued
            # keys (e.g. "acceptance_criteria": [...]), which would
            # otherwise be mistaken for the single-list-key wrapper.
            data = _flat_dict_to_proposals(data, anchor_pointers or [])
        elif _is_single_proposal(data):
            # The model gave up after proposing exactly ONE field but got
            # that proposal's own shape right -- {"field": "task_type",
            # "value": ..., "source": ..., "pointers": [...], ...} -- and
            # simply forgot to wrap it in a JSON array. Checked before the
            # generic wrapper-key heuristic below: that heuristic scans the
            # dict's own values for a list and would grab THIS object's
            # "pointers" (itself a list) as if it were the wrapped array,
            # turning a real, well-grounded proposal into a bare segment-id
            # string that verify_grounding() then silently drops -- an
            # observed cause of otherwise-inexplicable Level-4 abstains.
            data = [data]
        else:
            # Some local models wrap the array, e.g. {"proposals": [...]}
            list_vals = [v for v in data.values() if isinstance(v, list)]
            if len(list_vals) == 1:
                data = list_vals[0]
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of proposals")
    return data


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_grounding(items: list[dict], segments: list["architecture.Segment"]
                      ) -> list["architecture.FieldProposal"]:
    by_id = {s.segment_id: s for s in segments}
    proposals = []
    for item in items:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        if not isinstance(field, str):
            continue
        value = item.get("value")
        source = item.get("source", "UNKNOWN")
        pointers = [p for p in (item.get("pointers") or []) if p in by_id]
        evidence = item.get("evidence_text")
        confidence = item.get("confidence")

        if source != "UNKNOWN" and evidence:
            pointed_text = " ".join(by_id[p].text for p in pointers)
            if _normalize(evidence) not in _normalize(pointed_text):
                source = "INFERRED" if pointers else "UNKNOWN"
                confidence = min(confidence, 0.4) if isinstance(confidence, (int, float)) else 0.3
                if source == "UNKNOWN":
                    pointers, evidence, confidence = [], None, None

        proposals.append(architecture.FieldProposal(
            field=field, value=value, source=source, pointers=pointers,
            evidence_text=evidence, confidence=confidence, note=item.get("note"),
        ))
    return proposals


def default_fill_proposals(proposals: list["architecture.FieldProposal"]
                            ) -> list["architecture.FieldProposal"]:
    have = {p.field for p in proposals}
    filled = list(proposals)
    for f in LIST_FIELDS:
        if f not in have:
            filled.append(architecture.FieldProposal(f, [], "UNKNOWN"))
    if "dependencies" not in have:
        filled.append(architecture.FieldProposal("dependencies", [], "UNKNOWN"))
    if "scope" not in have:
        filled.append(architecture.FieldProposal("scope", {"in_scope": [], "out_of_scope": []}, "UNKNOWN"))
    if "acceptance_criteria" not in have:
        filled.append(architecture.FieldProposal("acceptance_criteria", [], "UNKNOWN"))
    for f in CLASSIFICATION_FIELDS:
        if f not in have:
            filled.append(architecture.FieldProposal(f, "Unknown", "UNKNOWN"))
    return filled


class OllamaEngine:
    """Satisfies the same engine.propose(segments) -> list[FieldProposal]
    contract as claude_label.py's ClaudeEngine, backed by a local model."""

    def __init__(self, model: str, base_url: str, timeout: int = 180, num_predict: int = 900):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.num_predict = num_predict
        # Set on every propose() call so the caller can log *why* a record
        # ended up abstained (Level 4) without a parse exception ever being
        # raised -- e.g. a properly-shaped response where every item came
        # back source="UNKNOWN" or with pointers that don't resolve. Without
        # this, a Level-4 abstain is a silent black box: nothing gets
        # written to debug_failed_responses (that path only fires on a
        # parse *exception*, not on a parse that succeeds but yields
        # nothing usable).
        self.last_raw: str = ""
        self.last_shape: str = ""
        self.last_n_items: int = 0

    def propose(self, segments: list["architecture.Segment"], weak: dict | None = None,
                 collection_id: str = "unknown") -> list["architecture.FieldProposal"]:
        user = (f"TEXT SEGMENTS:\n{render_segments(segments)}\n\n"
                f"PRE-FILL HINT (verify, do not trust blindly):\n{render_hint(weak)}")
        anchor_pointers = [s.segment_id for s in segments if s.type in ("ISSUE_TITLE", "ISSUE_BODY")]

        t0 = time.time()
        raw = call_ollama(SYSTEM_PROMPT, user, self.model, self.base_url,
                           timeout=self.timeout, num_predict=self.num_predict)
        elapsed = time.time() - t0

        data = None
        try:
            data = _parse_raw_json(raw)
        except (ValueError, json.JSONDecodeError):
            pass  # unparseable is also worth retrying, same as a flat response

        coverage = _covered_required_fields(data) if data is not None else set()
        sparse = (data is not None and not _is_flat_response(data)
                  and len(coverage) < MIN_REQUIRED_FIELDS_TO_ACCEPT)

        shape_label = "list"
        if data is None or _is_flat_response(data) or sparse:
            # First attempt gave the wrong shape (flat dict, nothing
            # parseable, or -- the case a plain shape-check misses -- a
            # syntactically fine response that only ever proposed a
            # handful of the 10 required fields). One retry with a
            # corrective nudge fixes this often enough with
            # qwen2.5:3b-instruct to be worth the extra call, rather than
            # silently accepting an ungrounded record on every issue.
            if data is None:
                reason = "unparseable response"
            elif _is_flat_response(data):
                reason = "flat response"
            else:
                reason = f"sparse response ({len(coverage)}/{len(REQUIRED_FIELD_NAMES)} required fields present)"
            print(f"  {reason} on attempt 1 -- retrying with corrective nudge", file=sys.stderr)

            t1 = time.time()
            raw_retry = call_ollama(SYSTEM_PROMPT, user + _CORRECTIVE_NUDGE, self.model,
                                     self.base_url, timeout=self.timeout, num_predict=self.num_predict)
            elapsed += time.time() - t1
            try:
                data_retry = _parse_raw_json(raw_retry)
            except (ValueError, json.JSONDecodeError):
                data_retry = None

            coverage_retry = _covered_required_fields(data_retry) if data_retry is not None else set()
            retry_is_better = (data_retry is not None and not _is_flat_response(data_retry)
                                and len(coverage_retry) > len(coverage))

            if retry_is_better:
                raw = raw_retry
                shape_label = "retry-recovered"
                print(f"  retry recovered {len(coverage_retry)}/{len(REQUIRED_FIELD_NAMES)} "
                      "required fields -- using it instead", file=sys.stderr)
            elif data is None or _is_flat_response(data):
                print("  retry did not fix it -- falling back to flat-recovery", file=sys.stderr)
                shape_label = "flat-recovered"
            else:
                print(f"  retry did not improve on the original ({len(coverage)}/"
                      f"{len(REQUIRED_FIELD_NAMES)} required fields) -- keeping it", file=sys.stderr)
                shape_label = "sparse-kept"

        print(f"  ({elapsed:.1f}s)", file=sys.stderr)

        try:
            items = parse_claude_response(raw, anchor_pointers)
        except (ValueError, json.JSONDecodeError) as e:
            # Save the raw text so we can see WHY parsing failed instead of
            # just losing it -- this is the single most useful debugging
            # signal when a local model doesn't follow the JSON contract.
            debug_dir = Path("data/debug_failed_responses")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{collection_id}.txt"
            debug_path.write_text(raw, encoding="utf-8")
            raise ValueError(f"{e} (raw response saved to {debug_path})") from e

        self.last_raw = raw
        self.last_n_items = len(items) if isinstance(items, list) else 0
        self.last_shape = shape_label

        proposals = verify_grounding(items, segments)
        return default_fill_proposals(proposals)


# --------------------------------------------------------- TrainingExample

def build_training_example(rec: dict, ground_truth: dict, license_: str, model_label: str) -> dict:
    core = rec["input_core"]
    src = ground_truth["task_identity"]["source_issue"]
    return {
        "example_id": str(uuid.uuid4()),
        "input": {
            "issue": {
                "title": core["title"], "body": core["body"],
                "labels": core["labels"], "comments": [c.get("body", "") for c in core["comments"]],
            },
            "context_tier": 1,
            "available_context": {},
        },
        "ground_truth": ground_truth,
        "label_provenance": {
            "labeling_method": "MODEL_ASSISTED_HUMAN_CORRECTED",
            "annotator_ids": [model_label],
            "source_repo": src["repo"],
            "source_issue_url": src["issue_url"],
            "snapshot_fetched_at": src["snapshot_fetched_at"],
            "license": license_,
            "labeled_at": architecture._now(),
            "labeling_tool_version": TOOL_VERSION,
        },
        "quality_status": {"tier": "SILVER", "quality_flags": [],
                            "reviewed_by": None, "review_notes": None},
        "collection_id": rec["collection_id"],
    }


def to_inference_input(rec: dict) -> "architecture.InferenceInput":
    core, ident = rec["input_core"], rec["identity"]
    return architecture.InferenceInput(
        repo=ident["source_repo"], issue_number=ident["issue_number"],
        issue_url=ident["source_issue_url"], snapshot_fetched_at=core["snapshot_fetched_at"],
        title=core["title"], body=core["body"], labels=core["labels"],
        comments=[{"author_role": "reporter" if i == 0 else "other", "body": c.get("body", ""),
                   "clarifies": False} for i, c in enumerate(core["comments"])],
        context_tier_ceiling=1,
    )


def record_license(rec: dict, fallback: str) -> str:
    return (rec.get("data_provenance") or {}).get("license") or fallback


# ------------------------------------------------------------- corrections
# (identical to claude_label.py -- kept so this script is a drop-in
# replacement, including for the --apply-corrections workflow)

def apply_corrections(out_path: Path, corrections_path: Path):
    examples = {}
    order = []
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            examples[ex["example_id"]] = ex
            order.append(ex["example_id"])

    n_applied = 0
    with corrections_path.open(encoding="utf-8") as fh:
        for line in fh:
            corr = json.loads(line)
            ex = examples.get(corr["example_id"])
            if ex is None:
                print(f"  skip: unknown example_id {corr['example_id']}", file=sys.stderr)
                continue
            gt = ex["ground_truth"]
            for field, value in corr.get("corrections", {}).items():
                if field in gt["task"]:
                    gt["task"][field] = value
                    gt["provenance"][field] = {
                        "source": "EXPLICIT",
                        "evidence": "human-corrected during Step 4 spot-check",
                        "confidence": 1.0,
                    }
                    ex["quality_status"]["quality_flags"] = list(
                        set(ex["quality_status"]["quality_flags"]) | {f"human_corrected:{field}"})
            annotator = corr.get("annotator_id")
            if annotator and annotator not in ex["label_provenance"]["annotator_ids"]:
                ex["label_provenance"]["annotator_ids"].append(annotator)
            ex["quality_status"]["reviewed_by"] = annotator or ex["quality_status"]["reviewed_by"]

            gt = finalize(gt)
            errors = validate_all(gt)
            if errors:
                print(f"  {ex['example_id']}: correction produced an invalid record, "
                      f"skipping write-back: {errors}", file=sys.stderr)
                continue
            ex["ground_truth"] = gt
            n_applied += 1

    with out_path.open("w", encoding="utf-8") as fh:
        for eid in order:
            fh.write(json.dumps(examples[eid], ensure_ascii=False) + "\n")
    print(f"applied {n_applied} correction(s) to {out_path}")


# --------------------------------------------------------------------- main

def load_weak_labels(path: Path) -> dict:
    weak = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                weak[row["collection_id"]] = row["weak_labels"]
    return weak


def iter_batched_records(records_by_id: dict, batches_path: Path):
    if batches_path.exists():
        with batches_path.open(encoding="utf-8") as fh:
            for line in fh:
                batch = json.loads(line)
                for cid in batch["collection_ids"]:
                    if cid in records_by_id:
                        yield cid
    else:
        yield from records_by_id.keys()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collected", default="data/collected.jsonl")
    ap.add_argument("--weak-labels", default="data/weak_labels.jsonl")
    ap.add_argument("--batches", default="data/labeling_batches.jsonl")
    ap.add_argument("--out", default="data/training_examples.jsonl")
    ap.add_argument("--spot-check-out", default="data/spot_check_needed.jsonl")
    ap.add_argument("--rejects-out", default="data/rejected.jsonl")
    ap.add_argument("--spot-check-rate", type=float, default=0.175)
    ap.add_argument("--model", default="llama3.1:8b", help="Ollama model tag, e.g. llama3.1:8b, qwen2.5:7b-instruct")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=180, help="seconds to wait for one issue's response")
    ap.add_argument("--num-predict", type=int, default=900, help="max tokens the model may generate per issue")
    ap.add_argument("--license", default="MIT", help="fallback if a record's own data_provenance.license is missing")
    ap.add_argument("--limit", type=int, default=None, help="label at most N issues (testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                     help="skip collection_ids already present in --out or --rejects-out, and "
                          "append instead of overwriting them. Use this for any run long enough "
                          "to risk being interrupted (a full-corpus pass on modest hardware can "
                          "take many hours) -- without it, a crash or closed terminal loses "
                          "every issue labeled so far, since --out is normally rewritten fresh "
                          "on each run.")
    ap.add_argument("--apply-corrections", default=None,
                     help="path to a corrections.jsonl; if given, all other labeling is skipped")
    args = ap.parse_args()

    if args.apply_corrections:
        apply_corrections(Path(args.out), Path(args.apply_corrections))
        return

    collected_path = Path(args.collected)
    records_by_id = {}
    with collected_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if is_labelable(rec):
                records_by_id[rec["collection_id"]] = rec
    if not records_by_id:
        sys.exit(f"No READY_FOR_LABELING records in {collected_path}. Run collect_issues.py first.")

    weak = load_weak_labels(Path(args.weak_labels))
    engine = OllamaEngine(args.model, args.ollama_url, timeout=args.timeout, num_predict=args.num_predict)
    rng = random.Random(args.seed)
    model_label = f"local:{args.model}"

    out_path, spot_path, rej_path = Path(args.out), Path(args.spot_check_out), Path(args.rejects_out)
    for p in (out_path, spot_path, rej_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    file_mode = "w"
    if args.resume:
        file_mode = "a"
        # Both --out (successes/abstains) and --rejects-out (hard failures)
        # represent an issue that's been dispositioned -- either counts as
        # "done" and should be skipped on resume, or a re-run would relabel
        # (and re-spend the 150-600s/issue compute cost) work already banked.
        for p in (out_path, rej_path):
            if p.exists():
                with p.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        cid = row.get("collection_id")
                        if cid:
                            already_done.add(cid)
        print(f"--resume: {len(already_done)} issue(s) already dispositioned -- skipping them",
              file=sys.stderr)

    n_written = n_rejected = n_spot = 0
    with out_path.open(file_mode, encoding="utf-8") as out_fh, \
         spot_path.open(file_mode, encoding="utf-8") as spot_fh, \
         rej_path.open(file_mode, encoding="utf-8") as rej_fh:

        for i, cid in enumerate(iter_batched_records(records_by_id, Path(args.batches))):
            if args.limit and i >= args.limit:
                break
            if cid in already_done:
                continue
            rec = records_by_id[cid]
            inp = to_inference_input(rec)
            task_identity = architecture._task_identity(inp)
            segments = architecture.gate(architecture.segment_issue(inp), inp.context_tier_ceiling)

            print(f"[{i + 1}] {cid} ...", file=sys.stderr)
            try:
                proposals = engine.propose(segments, weak.get(cid), collection_id=cid)
            except (ValueError, json.JSONDecodeError, requests.RequestException) as e:
                rej_fh.write(json.dumps({"collection_id": cid, "reason": f"engine_error: {e}"}) + "\n")
                n_rejected += 1
                continue

            ground_truth, trace = None, None
            try:
                ground_truth, trace = architecture.assemble(task_identity, proposals, segments)
            except Exception as e:
                # Defense in depth on top of the sanitizing in
                # _flat_dict_to_proposals: itu1/schema.py's enum check
                # (`value not in allowed_set`) raises TypeError instead of
                # a clean validation error on some malformed shapes, and
                # itu1/ is intentionally left unmodified (it's the tested
                # source of truth), so it can't be hardened directly here.
                # Losing one issue to an unexpected shape is fine; losing
                # the rest of the batch after it (and the compute already
                # spent on this one) is not.
                rej_fh.write(json.dumps({"collection_id": cid,
                                          "reason": f"assemble_crashed: {type(e).__name__}: {e}"}) + "\n")
                n_rejected += 1
                continue
            if isinstance(ground_truth, architecture.RejectedResult):
                rej_fh.write(json.dumps({"collection_id": cid, "reason_codes": ground_truth.reason_codes,
                                          "detail": ground_truth.detail}) + "\n")
                n_rejected += 1
                continue

            if getattr(trace, "degradation_level", 0) == 4:
                # A Level-4 abstain is schema-valid (all-Unknown), so it's
                # written to --out like a normal example, not --rejects-out
                # -- meaning nothing else in this script explains WHY it
                # abstained. Log it separately so that's diagnosable instead
                # of a silent "Unable to determine task from this issue".
                abstain_dir = Path("data/abstain_debug")
                abstain_dir.mkdir(parents=True, exist_ok=True)
                (abstain_dir / f"{cid}.json").write_text(json.dumps({
                    "collection_id": cid,
                    "response_shape": engine.last_shape,
                    "n_proposal_items": engine.last_n_items,
                    "trace_repairs": trace.repairs,
                    "trace_downgrades": trace.downgrades,
                    "raw_model_response": engine.last_raw,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

            try:
                lic = record_license(rec, args.license)
                ex = build_training_example(rec, ground_truth, lic, model_label)
                errors = example_mod.validate_example(ex)
            except Exception as e:
                # Same defense-in-depth rationale as the assemble() guard
                # above -- an unexpected shape here shouldn't cost the
                # rest of the batch either.
                rej_fh.write(json.dumps({"collection_id": cid,
                                          "reason": f"build_or_validate_crashed: {type(e).__name__}: {e}"}) + "\n")
                n_rejected += 1
                continue
            if errors:
                rej_fh.write(json.dumps({"collection_id": cid, "reason": "invalid_training_example",
                                          "errors": errors}) + "\n")
                n_rejected += 1
                continue

            out_fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_written += 1
            if rng.random() < args.spot_check_rate:
                spot_fh.write(json.dumps({
                    "example_id": ex["example_id"], "collection_id": cid,
                    "repo": rec["identity"]["source_repo"], "issue_number": rec["identity"]["issue_number"],
                    "issue_url": rec["identity"]["source_issue_url"],
                    "title": rec["input_core"]["title"],
                    "proposed": {f: ground_truth["task"].get(f) for f in
                                 (*CLASSIFICATION_FIELDS, "acceptance_criteria")},
                    "provenance": {f: ground_truth["provenance"].get(f) for f in CLASSIFICATION_FIELDS},
                    "review_required": ground_truth["review"]["review_required"],
                    "review_reasons": ground_truth["review"]["review_reasons"],
                }, ensure_ascii=False) + "\n")
                n_spot += 1

    session_label = "this session" if args.resume else "total"
    print(f"\n{session_label}: written {n_written}  rejected {n_rejected}  flagged for spot-check {n_spot}")
    if args.resume:
        print(f"cumulative (including prior sessions): "
              f"{len(already_done) + n_written + n_rejected} of {len(records_by_id)} issues dispositioned")
    print(f"-> {out_path}\n-> {spot_path}\n-> {rej_path}")
    if n_written:
        reject_rate = n_rejected / (n_written + n_rejected)
        print(f"\nreject rate: {reject_rate:.0%} -- if this is high (>25-30%), the local model is "
              "probably struggling with the JSON contract; try a bigger/more instruction-tuned model.")
    print(f"\nNext: review {spot_path} ({args.spot_check_rate:.0%} sample), write corrections to "
          f"a corrections.jsonl (one line: "
          '{"example_id": "...", "corrections": {"task_type": "Bug"}, "annotator_id": "you"}), then:\n'
          f"  python local_label.py --apply-corrections data/corrections.jsonl --out {out_path}\n"
          "The spot-check error rate you find also estimates the error rate for the unreviewed 80-85%.")


if __name__ == "__main__":
    main()