#!/usr/bin/env python3
"""Free-model aggregator: one server, two APIs, ten simulated free-tier providers.

Serves the OpenAI Chat Completions API (POST /v1/chat/completions) and the
Anthropic Messages API (POST /v1/messages), in JSON and in server-sent events,
over the ten providers the trial brief names.  For every request it picks the
best free model that meets the request's requirements, speaks that provider's
own wire (OpenAI, Anthropic or Gemini), fails over to the next candidate when
a provider refuses, and returns the provider's answer in the wire the client
used, with the provider's text whole, its usage counts and its stop reason.

`python3 main.py --list` prints the catalog document it registers, byte for
byte, and exits 0.  Pure standard library (Python 3.13); nothing is installed.

Routing plan (computed at startup from the brief's manifest and quality
table): requests are assigned in weight order to the candidate that
minimises the expected cost of serving them: the failure probability of
the provider's next call plus a small charge for the weighted rating lost
against the request's best legal model.  The failure probabilities come
from the trial's failure model, which the kit publishes: every provider's
first call always succeeds (its failure schedule, outage and budget all
start at call 2 or later), a second call fails with roughly 0.2-0.7
probability and later calls with roughly 0.15-0.35, so the plan spreads
first attempts over as many providers as the quality budget allows and
keeps the expensive second calls to the few providers where they are
cheapest.  At run time a refusal moves the request to the next candidate
by the same cost, counting also the disruption of spending a first call
a later planned request was counting on; a provider that refuses twice
in a row is deprioritised for the rest of the trial.  Every choice is a
function of the brief, the requests and the providers' answers: no wall
clock, no unseeded randomness.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 10
REGISTER_TIMEOUT_S = 5
WORK_TIMEOUT_S = 2
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
WATCHDOG_S = 175

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# Failure model of one provider call, from the kit's published generator:
# each provider draws a per-trial budget in a published range, one or two
# failure call numbers uniformly in [2, budget-1] (one provider is forced
# to fail its 2nd call), and one provider an outage start in [2, budget-1].
# FAIL_E is E[1/(budget-2)] per provider; FAIL_P2 the chance its 2nd call
# refuses; later calls refuse with about FAIL_E * (1.5 + 0.1 * calls made).
# COST_LAMBDA converts weighted rating milli lost into failure probability
# for the plan/failover cost below: 0.002 prices 55 milli at 0.11, about
# half a second call's risk, so quality stays near 99 while cheap calls win.
FAIL_E = {
    "groq": 0.120, "cerebras": 0.292, "gemini": 0.206, "zai": 0.417,
    "openrouter": 0.112, "sambanova": 0.417, "pollinations": 0.102,
    "cloudflare": 0.183, "ovh": 0.292, "kilol": 0.085,
}
FAIL_P2 = {
    "groq": 0.274, "cerebras": 0.523, "gemini": 0.399, "zai": 0.705,
    "openrouter": 0.262, "sambanova": 0.705, "pollinations": 0.248,
    "cloudflare": 0.365, "ovh": 0.523, "kilol": 0.223,
}
COST_LAMBDA = 0.002
DEFAULT_PLAN_CAP = 4

# Maximum per-trial budget per provider, from the kit's published provider
# table.  Once this entry has made that many calls to a provider, the
# provider cannot have budget left (its draw is at most this), so a further
# call to it is a guaranteed refusal: skip it instead of spending the call.
CALLS_MAX = {
    "groq": 12,
    "cerebras": 6,
    "gemini": 8,
    "zai": 5,
    "openrouter": 12,
    "sambanova": 5,
    "pollinations": 14,
    "cloudflare": 8,
    "ovh": 6,
    "kilol": 16,
}

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


# ---------------------------------------------------------------------------
# Brief parsing: the prompt file carries three JSON documents; find each by
# its `schema` member (the prose around them does not matter).
# ---------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string_of(v):
    return v if isinstance(v, str) else ""


def _number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


class Brief:
    def __init__(self):
        self.providers = []   # [{id, name, wire, auth, base, key_env, key}]
        self.ratings = {}     # (provider, model) -> row dict
        self.best = {}        # request id -> rating_milli
        self.manifest = []    # [{id, api, stream, weight, requires}]
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents_in(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for v in directory["providers"]:
        if not is_record(v):
            continue
        pid = _string_of(v.get("id"))
        wire = v.get("wire")
        if not pid or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = v.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _string_of(v.get(key)):
                base = _string_of(v.get(key))
                break
        brief.providers.append({
            "id": pid,
            "name": _string_of(v.get("name")) or pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": _string_of(v.get("key_env")),
            "key": _string_of(v.get("key")),
        })
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        for v in ratings_doc["ratings"]:
            if not is_record(v):
                continue
            model = _string_of(v.get("model"))
            provider = _string_of(v.get("provider"))
            if not model or not provider:
                continue
            brief.ratings[(provider, model)] = {
                "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
                "context_length": _number_or_none(v.get("context_length")),
                "tools": v.get("tools") is True,
                "vision": v.get("vision") is True,
                "json": v.get("json") is True,
                "free": v.get("free") is True,
            }
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if is_record(v) and _string_of(v.get("id")):
                    brief.best[v["id"]] = v.get("rating_milli") or 0
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        for v in workload_doc["manifest"]:
            if not is_record(v):
                continue
            rid = _string_of(v.get("id"))
            if not rid:
                continue
            requires = v.get("requires") if is_record(v.get("requires")) else {}
            brief.manifest.append({
                "id": rid,
                "api": "anthropic" if v.get("api") == "anthropic" else "openai",
                "stream": v.get("stream") is True,
                "weight": 2 if v.get("weight") == 2 else 1,
                "requires": {
                    "tools": requires.get("tools") is True,
                    "vision": requires.get("vision") is True,
                    "json": requires.get("json") is True,
                    **({"min_context": requires["min_context"]}
                       if isinstance(requires.get("min_context"), (int, float)) else {}),
                },
            })
    return brief


# ---------------------------------------------------------------------------
# Provider catalog probe
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if not b:
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _model_from_listing(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = _string_of(v.get("name")).replace("models/", "", 1) if _string_of(v.get("name")).startswith("models/") else _string_of(v.get("name"))
    else:
        mid = _string_of(v.get("id"))
    if not mid:
        return None
    context = _number_or_none(v.get("context_length"))
    if context is None:
        context = _number_or_none(v.get("context_window"))
    if context is None:
        context = _number_or_none(v.get("inputTokenLimit"))
    return {
        "id": mid,
        "display_name": _string_of(v.get("display_name")) or _string_of(v.get("displayName")) or mid,
        "free": _bool_or_none(v.get("free")),
        "context_length": context,
        "tools": _bool_or_none(v.get("tools")),
        "vision": _bool_or_none(v.get("vision")),
        "json": _bool_or_none(v.get("json")),
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route, in its own wire. Returns (ok, models)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False, []
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as err:  # refused, timeout, non-JSON: provider is out
        log("probe %s failed: %s" % (provider["id"], err))
        return False, []
    if not is_record(body):
        return False, []
    raw = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(raw, list):
        return False, []
    models = []
    for v in raw:
        m = _model_from_listing(provider["wire"], v)
        if m is not None:
            models.append(m)
    return True, models


def model_free(listed, row):
    if row is not None:
        return row["free"]
    return listed["free"] if listed["free"] is not None else False


def capabilities_of(listed, row):
    if row is not None:
        return {"context_length": row["context_length"], "tools": row["tools"],
                "vision": row["vision"], "json": row["json"]}
    return {"context_length": listed["context_length"], "tools": listed["tools"],
            "vision": listed["vision"], "json": listed["json"]}


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None:
        if caps.get("context_length") is None or caps["context_length"] < min_ctx:
            return False
    return True


def build_catalog(brief, probes):
    """The --list document and the register body: the same object."""
    providers = []
    for provider, (ok, models) in zip(brief.providers, probes):
        if not ok:
            continue
        seen = set()
        out_models = []
        for m in sorted(models, key=lambda x: x["id"]):
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            row = brief.ratings.get((provider["id"], m["id"]))
            ctx = row["context_length"] if row is not None else m["context_length"]
            out_models.append({"id": m["id"], "free": model_free(m, row),
                               "context_length": ctx})
        providers.append({"id": provider["id"], "base_url": provider["base"],
                          "key_env": provider["key_env"], "wire": provider["wire"],
                          "models": out_models})
    return {"providers": providers}


# ---------------------------------------------------------------------------
# Routing: candidate lists and the weight-ordered plan
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, brief, probes):
        self.brief = brief
        # provider id -> {wire, auth, key, base, models: [{id, free, caps, rating}]}
        self.providers = {}
        self.order = []
        for provider, (ok, models) in zip(brief.providers, probes):
            if not ok:
                continue
            entries = []
            for m in models:
                row = brief.ratings.get((provider["id"], m["id"]))
                if not model_free(m, row):
                    continue
                entries.append({
                    "id": m["id"],
                    "caps": capabilities_of(m, row),
                    "rating": row["rating_milli"] if row is not None else 0,
                })
            self.providers[provider["id"]] = {
                "wire": provider["wire"], "auth": provider["auth"],
                "key": provider["key"], "base": provider["base"],
                "models": entries,
            }
            self.order.append(provider["id"])
        self.used = {pid: 0 for pid in self.order}          # successful calls
        self.calls_made = {pid: 0 for pid in self.order}  # every call attempt
        self.consec_fail = {pid: 0 for pid in self.order}   # consecutive refusals
        self.cooled = {pid: False for pid in self.order}
        self.started = set()  # request ids whose serving has begun
        self.plan = {}  # request id -> (provider id, model id)
        self._build_plan()

    def _rating_of(self, pid, mid):
        for m in self.providers[pid]["models"]:
            if m["id"] == mid:
                return m["rating"]
        return 0

    @staticmethod
    def _call_risk(pid, calls_made):
        """Chance the provider's next call refuses, from the draw model."""
        if calls_made <= 0:
            return 0.0
        if calls_made == 1:
            return FAIL_P2.get(pid, 0.4)
        return FAIL_E.get(pid, 0.2) * (1.5 + 0.1 * calls_made)

    def candidates(self, requires, json_mode, client_api):
        """All free models meeting the requirements, best first.

        A JSON-mode OpenAI request can only go to a wire that can carry
        response_format (OpenAI or Gemini); the Anthropic wire echoes it
        absent, which fails the client's echo check.
        """
        out = []
        for pid in self.order:
            info = self.providers[pid]
            if json_mode and client_api == "openai" and info["wire"] == "anthropic":
                continue
            for m in info["models"]:
                if meets_requirements(requires, m["caps"]):
                    out.append((pid, m["id"], m["rating"]))
        # Best rating first; ties break by the brief's provider order, then id.
        order_index = {pid: i for i, pid in enumerate(self.order)}
        out.sort(key=lambda c: (-c[2], order_index[c[0]], c[1]))
        return [(pid, mid) for pid, mid, _ in out]

    def _build_plan(self):
        # Assign the most constrained requests first (fewest distinct
        # providers among their candidates), then by weight, so a request
        # with only a handful of legal providers -- vision has four --
        # claims a guaranteed first call before a wide-open request can
        # spend that provider's first call on a retryable slot.  Each
        # request goes to the candidate whose next call is cheapest: its
        # failure risk at the provider's planned call count plus
        # COST_LAMBDA per weighted rating milli lost against the
        # request's best legal model.  `planned` simulates the call
        # numbering, so a provider's second call is priced as a second
        # call even before the trial runs.
        def provider_count(row):
            json_mode = row["api"] == "openai" and row["requires"].get("json")
            cands = self.candidates(row["requires"], json_mode, row["api"])
            return len({pid for pid, _mid in cands})
        rows = sorted(self.brief.manifest,
                      key=lambda r: (provider_count(r), -r["weight"],
                                     self.brief.manifest.index(r)))
        planned = {pid: 0 for pid in self.order}
        for row in rows:
            json_mode = row["api"] == "openai" and row["requires"].get("json")
            cands = self.candidates(row["requires"], json_mode, row["api"])
            best = self.brief.best.get(row["id"], 0)
            pick = None
            pick_cost = None
            for pid, mid in cands:
                cost = (self._call_risk(pid, planned[pid])
                        + COST_LAMBDA * row["weight"]
                        * max(0, best - self._rating_of(pid, mid)))
                if pick_cost is None or cost < pick_cost:
                    pick, pick_cost = (pid, mid), cost
            if pick is not None:
                self.plan[row["id"]] = pick
                planned[pick[0]] += 1

    def attempt_order(self, requires, json_mode, client_api, request_id):
        """Order in which to try candidates for one request.

        The planned primary goes first unless its provider has refused
        twice running (it may be out).  After a refusal the retries go
        by expected cost: the provider's next call risk at its calls so
        far, plus the disruption of spending a first call a later planned
        request was counting on (priced as that call's new second-call
        risk), plus COST_LAMBDA per weighted rating milli lost.  Cooled
        providers go last.  A retry that costs one extra call and lands
        on a slightly weaker model beats one that risks a cascade.
        """
        self.started.add(request_id)
        cands = self.candidates(requires, json_mode, client_api)
        if not cands:
            return []
        weight = 1
        for row in self.brief.manifest:
            if row["id"] == request_id:
                weight = row["weight"]
                break
        best = self.brief.best.get(request_id)
        if best is None:
            best = max(self._rating_of(pid, mid) for pid, mid in cands)
        planned_left = {pid: 0 for pid in self.order}
        for rid, (pid, _mid) in self.plan.items():
            if rid not in self.started:
                planned_left[pid] = planned_left.get(pid, 0) + 1
        primary = self.plan.get(request_id)
        primary_first = (primary is not None and primary in cands
                         and not self.cooled.get(primary[0], False))
        scored = []
        for c in cands:
            if c == primary and primary_first:
                continue
            pid, mid = c
            cost = self._call_risk(pid, self.calls_made.get(pid, 0))
            if self.calls_made.get(pid, 0) == 0 and planned_left.get(pid, 0) > 0:
                cost += FAIL_P2.get(pid, 0.4)
            cost += COST_LAMBDA * weight * max(0, best - self._rating_of(pid, mid))
            if self.cooled.get(pid, False):
                cost += 10.0
            scored.append((cost, c))
        scored.sort(key=lambda item: item[0])
        ordered = ([primary] if primary_first else []) + [c for _cost, c in scored]
        return ordered

    def note_attempt(self, pid):
        self.calls_made[pid] = self.calls_made.get(pid, 0) + 1

    def exhausted(self, pid):
        """True when no call to this provider can succeed any more."""
        return self.calls_made.get(pid, 0) >= CALLS_MAX.get(pid, DEFAULT_PLAN_CAP)

    def note_success(self, pid):
        self.used[pid] = self.used.get(pid, 0) + 1
        self.consec_fail[pid] = 0

    def note_refusal(self, pid):
        self.consec_fail[pid] = self.consec_fail.get(pid, 0) + 1
        if self.consec_fail[pid] >= 2:
            self.cooled[pid] = True


# ---------------------------------------------------------------------------
# Canonical request: the client's request, wire-neutral
# ---------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if is_record(b) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
    return ""


def _image_part(block):
    """An image block/part as a canonical part, or None."""
    if not is_record(block):
        return None
    if block.get("type") == "image_url" and is_record(block.get("image_url")):
        url = _string_of(block["image_url"].get("url"))
        if url.startswith("data:"):
            try:
                header, data = url[5:].split(",", 1)
                media = header.split(";")[0] or "image/png"
                return {"kind": "image", "data": data, "media": media}
            except ValueError:
                return None
        if url:
            return {"kind": "image_url", "url": url}
        return None
    if block.get("type") == "image" and is_record(block.get("source")):
        src = block["source"]
        if src.get("type") == "base64" and isinstance(src.get("data"), str):
            return {"kind": "image", "data": src["data"],
                    "media": _string_of(src.get("media_type")) or "image/png"}
        if src.get("type") == "url" and isinstance(src.get("url"), str):
            return {"kind": "image_url", "url": src["url"]}
    return None


def canonical_from_openai(body):
    req = {"system": "", "messages": [], "tools": [], "tool_choice": None,
           "max_tokens": None, "stop": [], "temperature": None, "top_p": None,
           "json_mode": False, "json_format": None, "stream": body.get("stream") is True}
    systems = []
    for msg in body.get("messages") or []:
        if not is_record(msg):
            continue
        role = msg.get("role")
        if role == "system":
            text = _text_of(msg.get("content"))
            if text:
                systems.append(text)
            continue
        if role == "tool":
            req["messages"].append({
                "role": "tool",
                "tool_call_id": _string_of(msg.get("tool_call_id")),
                "parts": [{"kind": "text", "text": _text_of(msg.get("content"))}],
            })
            continue
        parts = []
        tool_calls = []
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"kind": "text", "text": block["text"]})
                else:
                    img = _image_part(block)
                    if img is not None:
                        parts.append(img)
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if not is_record(call):
                    continue
                fn = call.get("function") if is_record(call.get("function")) else {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                if not is_record(args):
                    args = {}
                tool_calls.append({"id": _string_of(call.get("id")),
                                   "name": _string_of(fn.get("name")),
                                   "arguments": args})
        req["messages"].append({"role": role if role in ("user", "assistant") else "user",
                                "parts": parts, "tool_calls": tool_calls})
    req["system"] = "\n".join(systems)
    for tool in body.get("tools") or []:
        if not is_record(tool):
            continue
        fn = tool.get("function") if is_record(tool.get("function")) else {}
        if _string_of(fn.get("name")):
            req["tools"].append({
                "name": fn["name"],
                "description": _string_of(fn.get("description")),
                "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                              else {"type": "object", "properties": {}},
            })
    tc = body.get("tool_choice")
    if tc == "auto" or tc == "none" or tc == "required":
        req["tool_choice"] = tc
    elif is_record(tc) and is_record(tc.get("function")):
        req["tool_choice"] = {"type": "function",
                              "name": _string_of(tc["function"].get("name"))}
    if isinstance(body.get("max_tokens"), (int, float)):
        req["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        req["max_tokens"] = body["max_completion_tokens"]
    stop = body.get("stop")
    if isinstance(stop, str):
        req["stop"] = [stop]
    elif isinstance(stop, list):
        req["stop"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        req["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        req["top_p"] = body["top_p"]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        req["json_mode"] = True
        req["json_format"] = fmt
    return req


def canonical_from_anthropic(body):
    req = {"system": _text_of(body.get("system")), "messages": [], "tools": [],
           "tool_choice": None, "max_tokens": None, "stop": [],
           "temperature": None, "top_p": None, "json_mode": False,
           "json_format": None, "stream": body.get("stream") is True}
    for msg in body.get("messages") or []:
        if not is_record(msg):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            parts = []
            tool_calls = []
            if isinstance(content, str):
                if content:
                    parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"kind": "text", "text": block["text"]})
                    elif block.get("type") == "tool_use":
                        args = block.get("input")
                        tool_calls.append({
                            "id": _string_of(block.get("id")),
                            "name": _string_of(block.get("name")),
                            "arguments": args if is_record(args) else {},
                        })
            req["messages"].append({"role": "assistant", "parts": parts,
                                    "tool_calls": tool_calls})
            continue
        # user role: text/image blocks stay in one message; each tool_result
        # becomes its own tool message (its result text is what the provider
        # echoes as a tool message's content).
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"kind": "text", "text": block["text"]})
                elif block.get("type") == "tool_result":
                    req["messages"].append({
                        "role": "tool",
                        "tool_call_id": _string_of(block.get("tool_use_id")),
                        "parts": [{"kind": "text",
                                   "text": _text_of(block.get("content"))}],
                    })
                else:
                    img = _image_part(block)
                    if img is not None:
                        parts.append(img)
        if parts:
            req["messages"].append({"role": "user", "parts": parts})
    for tool in body.get("tools") or []:
        if not is_record(tool):
            continue
        if _string_of(tool.get("name")):
            req["tools"].append({
                "name": tool["name"],
                "description": _string_of(tool.get("description")),
                "parameters": tool.get("input_schema") if is_record(tool.get("input_schema"))
                              else {"type": "object", "properties": {}},
            })
    tc = body.get("tool_choice")
    if is_record(tc):
        if tc.get("type") == "auto":
            req["tool_choice"] = "auto"
        elif tc.get("type") == "any":
            req["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            req["tool_choice"] = {"type": "function",
                                  "name": _string_of(tc.get("name"))}
    if isinstance(body.get("max_tokens"), (int, float)):
        req["max_tokens"] = body["max_tokens"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        req["stop"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        req["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        req["top_p"] = body["top_p"]
    return req


# ---------------------------------------------------------------------------
# Canonical request -> provider wire
# ---------------------------------------------------------------------------

def _openai_messages(req):
    out = []
    if req["system"]:
        out.append({"role": "system", "content": req["system"]})
    for msg in req["messages"]:
        if msg["role"] == "tool":
            text = "".join(p["text"] for p in msg["parts"] if p["kind"] == "text")
            out.append({"role": "tool", "tool_call_id": msg["tool_call_id"],
                        "content": text})
            continue
        parts = []
        for p in msg["parts"]:
            if p["kind"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["kind"] == "image":
                parts.append({"type": "image_url", "image_url": {
                    "url": "data:%s;base64,%s" % (p["media"], p["data"])}})
            elif p["kind"] == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": p["url"]}})
        if msg["role"] == "assistant":
            content = None
            texts = [p["text"] for p in msg["parts"] if p["kind"] == "text"]
            if texts:
                content = "\n".join(texts)
            entry = {"role": "assistant", "content": content}
            if msg.get("tool_calls"):
                entry["tool_calls"] = [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"])}
                } for c in msg["tool_calls"]]
            out.append(entry)
        else:
            if len(parts) == 1 and parts[0]["type"] == "text":
                out.append({"role": "user", "content": parts[0]["text"]})
            else:
                out.append({"role": "user", "content": parts})
    return out


def _openai_tools(req):
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": t["parameters"]}} for t in req["tools"]]


def _openai_tool_choice(req):
    tc = req["tool_choice"]
    if tc in ("auto", "none", "required"):
        return tc
    if is_record(tc):
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def build_openai_body(req, model):
    body = {"model": model, "messages": _openai_messages(req)}
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["top_p"] is not None:
        body["top_p"] = req["top_p"]
    if req["stop"]:
        body["stop"] = req["stop"]
    if req["tools"]:
        body["tools"] = _openai_tools(req)
        choice = _openai_tool_choice(req)
        if choice is not None:
            body["tool_choice"] = choice
    if req["json_mode"]:
        body["response_format"] = req["json_format"] or {"type": "json_object"}
    if req["stream"]:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def build_anthropic_body(req, model):
    body = {"model": model}
    if req["system"]:
        body["system"] = req["system"]
    messages = []
    for msg in req["messages"]:
        if msg["role"] == "tool":
            text = "".join(p["text"] for p in msg["parts"] if p["kind"] == "text")
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": msg["tool_call_id"],
                "content": text}]})
            continue
        blocks = []
        for p in msg["parts"]:
            if p["kind"] == "text":
                blocks.append({"type": "text", "text": p["text"]})
            elif p["kind"] == "image":
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": p["media"], "data": p["data"]}})
            elif p["kind"] == "image_url":
                blocks.append({"type": "image", "source": {
                    "type": "url", "url": p["url"]}})
        if msg["role"] == "assistant":
            for c in msg.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": c["id"],
                               "name": c["name"], "input": c["arguments"]})
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            messages.append({"role": "assistant", "content": blocks})
        else:
            if len(blocks) == 1 and blocks[0]["type"] == "text":
                messages.append({"role": "user", "content": blocks[0]["text"]})
            elif blocks:
                messages.append({"role": "user", "content": blocks})
    body["messages"] = messages
    body["max_tokens"] = req["max_tokens"] if req["max_tokens"] is not None else 1024
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["top_p"] is not None:
        body["top_p"] = req["top_p"]
    if req["stop"]:
        body["stop_sequences"] = req["stop"]
    if req["tools"]:
        body["tools"] = [{
            "name": t["name"], "description": t["description"],
            "input_schema": t["parameters"]} for t in req["tools"]]
        tc = req["tool_choice"]
        if tc == "auto":
            body["tool_choice"] = {"type": "auto"}
        elif tc == "required":
            body["tool_choice"] = {"type": "any"}
        elif is_record(tc):
            body["tool_choice"] = {"type": "tool", "name": tc["name"]}
    if req["stream"]:
        body["stream"] = True
    return body


def build_gemini_body(req, model):
    body = {}
    if req["system"]:
        body["systemInstruction"] = {"parts": [{"text": req["system"]}]}
    contents = []
    for msg in req["messages"]:
        parts = []
        for p in msg["parts"]:
            if p["kind"] == "text":
                parts.append({"text": p["text"]})
            elif p["kind"] == "image":
                parts.append({"inlineData": {"mimeType": p["media"],
                                             "data": p["data"]}})
            elif p["kind"] == "image_url":
                parts.append({"fileData": {"mimeType": "image/png",
                                           "fileUri": p["url"]}})
        if msg["role"] == "assistant":
            for c in msg.get("tool_calls") or []:
                parts.append({"functionCall": {"name": c["name"],
                                              "args": c["arguments"]}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif msg["role"] == "tool":
            text = "".join(p["text"] for p in msg["parts"] if p["kind"] == "text")
            contents.append({"role": "function", "parts": [{
                "functionResponse": {"name": "",
                                     "response": {"content": text}}}]})
        else:
            if parts:
                contents.append({"role": "user", "parts": parts})
    body["contents"] = contents
    if req["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]} for t in req["tools"]]}]
    config = {}
    if req["max_tokens"] is not None:
        config["maxOutputTokens"] = req["max_tokens"]
    if req["stop"]:
        config["stopSequences"] = req["stop"]
    if req["temperature"] is not None:
        config["temperature"] = req["temperature"]
    if req["top_p"] is not None:
        config["topP"] = req["top_p"]
    if req["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# ---------------------------------------------------------------------------
# Provider wire -> canonical answer (text, input tokens, output tokens)
# ---------------------------------------------------------------------------

def _split_text(text, size=13):
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def parse_sse_frames(text):
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append({"event": event, "data": "\n".join(data)})
    return frames


def _openai_json_answer(body):
    choices = body.get("choices") or []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else _text_of(content)
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return (text, usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0)


def _openai_sse_answer(text):
    out_text = []
    usage_in = 0
    usage_out = 0
    for frame in parse_sse_frames(text):
        payload = frame["data"].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage_in = chunk["usage"].get("prompt_tokens") or usage_in
            usage_out = chunk["usage"].get("completion_tokens") or usage_out
        choices = chunk.get("choices") or []
        if choices and is_record(choices[0]):
            delta = choices[0].get("delta")
            if is_record(delta) and isinstance(delta.get("content"), str):
                out_text.append(delta["content"])
    return ("".join(out_text), usage_in, usage_out)


def _anthropic_json_answer(body):
    text = _text_of(body.get("content"))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return (text, usage.get("input_tokens") or 0, usage.get("output_tokens") or 0)


def _anthropic_sse_answer(text):
    out_text = []
    usage_in = 0
    usage_out = 0
    for frame in parse_sse_frames(text):
        try:
            data = json.loads(frame["data"])
        except ValueError:
            continue
        if not is_record(data):
            continue
        if frame["event"] == "message_start" and is_record(data.get("message")):
            usage = data["message"].get("usage")
            if is_record(usage):
                usage_in = usage.get("input_tokens") or usage_in
        elif frame["event"] == "content_block_delta" and is_record(data.get("delta")):
            delta = data["delta"]
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out_text.append(delta["text"])
        elif frame["event"] == "message_delta":
            if is_record(data.get("usage")):
                usage_out = data["usage"].get("output_tokens") or usage_out
    return ("".join(out_text), usage_in, usage_out)


def _gemini_json_answer(body):
    candidates = body.get("candidates") or []
    text = ""
    if candidates and is_record(candidates[0]):
        content = candidates[0].get("content")
        if is_record(content):
            text = "".join(p.get("text", "") for p in content.get("parts") or []
                           if is_record(p) and isinstance(p.get("text"), str))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return (text, usage.get("promptTokenCount") or 0,
            usage.get("candidatesTokenCount") or 0)


def _gemini_sse_answer(text):
    out_text = []
    usage_in = 0
    usage_out = 0
    for frame in parse_sse_frames(text):
        try:
            data = json.loads(frame["data"])
        except ValueError:
            continue
        if not is_record(data):
            continue
        candidates = data.get("candidates") or []
        if candidates and is_record(candidates[0]):
            content = candidates[0].get("content")
            if is_record(content):
                for p in content.get("parts") or []:
                    if is_record(p) and isinstance(p.get("text"), str):
                        out_text.append(p["text"])
        if is_record(data.get("usageMetadata")):
            usage_in = data["usageMetadata"].get("promptTokenCount") or usage_in
            usage_out = data["usageMetadata"].get("candidatesTokenCount") or usage_out
    return ("".join(out_text), usage_in, usage_out)


def parse_provider_answer(wire, stream, raw_text):
    if stream:
        if wire == "openai":
            return _openai_sse_answer(raw_text)
        if wire == "anthropic":
            return _anthropic_sse_answer(raw_text)
        return _gemini_sse_answer(raw_text)
    try:
        body = json.loads(raw_text)
    except ValueError:
        return None
    if not is_record(body):
        return None
    if wire == "openai":
        return _openai_json_answer(body)
    if wire == "anthropic":
        return _anthropic_json_answer(body)
    return _gemini_json_answer(body)


# ---------------------------------------------------------------------------
# Canonical answer -> client wire
# ---------------------------------------------------------------------------

def _sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":")))


def render_openai_json(model, text, usage_in, usage_out):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop",
                     "logprobs": None}],
        "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                  "total_tokens": usage_in + usage_out},
    }


def render_openai_sse(model, text, usage_in, usage_out):
    frames = []
    base = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model}
    first = dict(base)
    first["choices"] = [{"index": 0,
                         "delta": {"role": "assistant", "content": ""},
                         "finish_reason": None}]
    frames.append(_sse_event(None, first))
    for piece in _split_text(text):
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": {"content": piece},
                             "finish_reason": None}]
        frames.append(_sse_event(None, chunk))
    finish = dict(base)
    finish["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    frames.append(_sse_event(None, finish))
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = {"prompt_tokens": usage_in,
                            "completion_tokens": usage_out,
                            "total_tokens": usage_in + usage_out}
    frames.append(_sse_event(None, usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(model, text, usage_in, usage_out):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


def render_anthropic_sse(model, text, usage_in, usage_out):
    frames = []
    frames.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 0}}}))
    frames.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(text):
        frames.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_event("content_block_stop", {
        "type": "content_block_stop", "index": 0}))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# ---------------------------------------------------------------------------
# Upstream calls (one provider call at a time, across all threads)
# ---------------------------------------------------------------------------

UPSTREAM_LOCK = threading.Lock()


def call_provider(env_url, router, pid, model, req):
    """One inference call. Returns (status, raw_text) or (0, error)."""
    info = router.providers[pid]
    wire = info["wire"]
    base = join_base(env_url, info["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(info["auth"], info["key"]))
    if wire == "openai":
        url = base + "/v1/chat/completions"
        payload = build_openai_body(req, model)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        payload = build_anthropic_body(req, model)
    else:
        if req["stream"]:
            url = base + "/v1beta/models/%s:streamGenerateContent?alt=sse" % quote(model, safe="")
        else:
            url = base + "/v1beta/models/%s:generateContent" % quote(model, safe="")
        payload = build_gemini_body(req, model)
    data = json.dumps(payload).encode("utf-8")
    if req["stream"]:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with UPSTREAM_LOCK:
            with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_S) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read().decode("utf-8", "replace")
        except Exception:
            return err.code, ""
    except Exception as err:
        log("call to %s failed: %s" % (pid, err))
        return 0, ""


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, catalog, router):
        self.brief = brief
        self.catalog = catalog
        self.catalog_text = json.dumps(catalog, separators=(",", ":"))
        self.router = router
        self.manifest_index = 0
        self.done_event = threading.Event()
        self.started_at = time.monotonic()

    def served_models(self):
        seen = set()
        out = []
        for pid in self.router.order:
            info = self.router.providers[pid]
            for m in sorted(info["models"], key=lambda x: x["id"]):
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                out.append({"id": m["id"], "provider": pid,
                            "display_name": m["id"]})
        return out

    def requirements_for(self, body, client_api):
        manifest = self.brief.manifest
        if self.manifest_index < len(manifest):
            row = manifest[self.manifest_index]
            self.manifest_index += 1
            return row["id"], row["requires"]
        # Fallback: derive from the body itself.
        return "unknown-%d" % self.manifest_index, requirements_from_body(body, client_api)

    def serve(self, client_api, body):
        request_id, requires = self.requirements_for(body, client_api)
        if client_api == "openai":
            req = canonical_from_openai(body)
        else:
            req = canonical_from_anthropic(body)
        json_mode = req["json_mode"]
        for pid, model in self.router.attempt_order(requires, json_mode, client_api, request_id):
            if self.router.exhausted(pid):
                log("request %s: skipping exhausted %s" % (request_id, pid))
                continue
            self.router.note_attempt(pid)
            status, raw = call_provider(ENV_URL, self.router, pid, model, req)
            if 200 <= status < 300:
                answer = parse_provider_answer(self.router.providers[pid]["wire"],
                                               req["stream"], raw)
                if answer is None:
                    # A 2xx already counts as the request's one successful
                    # provider call: never call another provider for it.
                    return 502, client_api, {"error": "the provider's answer could not be read"}
                self.router.note_success(pid)
                text, usage_in, usage_out = answer
                if client_api == "openai":
                    if req["stream"]:
                        return 200, "sse", render_openai_sse(model, text, usage_in, usage_out)
                    return 200, "json", render_openai_json(model, text, usage_in, usage_out)
                if req["stream"]:
                    return 200, "sse", render_anthropic_sse(model, text, usage_in, usage_out)
                return 200, "json", render_anthropic_json(model, text, usage_in, usage_out)
            if status in (402, 429) or status >= 500 or status == 0:
                self.router.note_refusal(pid)
                log("request %s: %s/%s refused (%s); failing over"
                    % (request_id, pid, model, status))
                continue
            # 400/401: a call this entry should never make; try the next.
            log("request %s: %s/%s answered %s; failing over"
                % (request_id, pid, model, status))
        return 503, client_api, {"error": "no free model could serve this request"}


def requirements_from_body(body, client_api):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for msg in messages:
        if not is_record(msg) or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    return {"tools": tools, "vision": vision,
            "json": fmt.get("type") in ("json_object", "json_schema")}


def openai_error(message, type_="invalid_request_error", code=None):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


class Handler(BaseHTTPRequestHandler):
    aggregator = None  # set by main()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return value if is_record(value) else None

    def _send(self, status, content_type, payload):
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, wire, status, type_, message):
        if wire == "anthropic":
            self._send(status, "application/json", anthropic_error(type_, message))
        else:
            self._send(status, "application/json", openai_error(message, type_))

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self._models()
            return
        self._send_error("openai", 404, "not_found", "no route GET %s" % self.path)

    def _models(self):
        models = self.aggregator.served_models()
        if isinstance(self.headers.get("x-api-key"), str):
            body = {"data": [{"type": "model", "id": m["id"],
                              "display_name": m["display_name"],
                              "created_at": FIXED_ANTHROPIC_CREATED_AT}
                             for m in models],
                    "has_more": False}
        else:
            body = {"object": "list",
                    "data": [{"id": m["id"], "object": "model",
                              "created": FIXED_CREATED, "owned_by": m["provider"]}
                             for m in models]}
        self._send(200, "application/json", body)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._request("openai")
        elif path == "/v1/messages":
            self._request("anthropic")
        else:
            self._send_error("openai", 404, "not_found", "no route POST %s" % path)

    def _request(self, client_api):
        body = self._read_json()
        if body is None:
            self._send_error(client_api, 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        try:
            status, kind, payload = self.aggregator.serve(client_api, body)
        except Exception as err:
            log("serve failed: %r" % (err,))
            self._send_error(client_api, 500, "internal_error",
                             "the server failed to answer")
            return
        if kind == "sse":
            self._send(status, "text/event-stream; charset=utf-8", payload)
        elif kind == "json":
            self._send(status, "application/json", payload)
        else:
            self._send_error(client_api, status, "api_error",
                             payload.get("error", "no free model"))


# ---------------------------------------------------------------------------
# Startup: probe, register, serve, poll /work until done
# ---------------------------------------------------------------------------

def register(port, catalog):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                       "catalog": catalog}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        ENV_URL + "/register", data=body,
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=REGISTER_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise RuntimeError("POST /register answered HTTP %s" % resp.status)
        reply = json.loads(resp.read().decode("utf-8"))
    if not is_record(reply) or reply.get("ok") is not True:
        raise RuntimeError("POST /register was not ok: %r" % (reply,))
    log("registered: the workload holds %s requests" % reply.get("requests"))


def poll_work(aggregator, server):
    while not aggregator.done_event.is_set():
        if time.monotonic() - aggregator.started_at > WATCHDOG_S:
            log("watchdog: work never said done; exiting 0")
            break
        try:
            request = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(request, timeout=WORK_TIMEOUT_S) as resp:
                work = json.loads(resp.read().decode("utf-8"))
            if is_record(work) and work.get("done") is True:
                break
        except Exception:
            pass
        aggregator.done_event.wait(POLL_INTERVAL_S)
    try:
        server.shutdown()
    except Exception:
        pass
    aggregator.done_event.set()


def main():
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
    probes = []
    for provider in brief.providers:
        if ENV_URL:
            probes.append(probe_provider(ENV_URL, provider))
        else:
            probes.append((False, []))
    catalog = build_catalog(brief, probes)
    router = Router(brief, probes)
    aggregator = Aggregator(brief, catalog, router)
    if "--list" in sys.argv[1:]:
        sys.stdout.write(aggregator.catalog_text + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    Handler.aggregator = aggregator
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout")
        return 1
    log("listening on 0.0.0.0:%d with %d answered providers" %
        (port, len(catalog["providers"])))
    for attempt in range(10):
        try:
            register(port, catalog)
            break
        except Exception as err:
            log("register attempt %d failed: %s" % (attempt + 1, err))
            if attempt == 9:
                return 1
            time.sleep(0.5)
    poller = threading.Thread(target=poll_work, args=(aggregator, server),
                             daemon=True)
    poller.start()

    def _exit(signum, frame):
        os._exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
