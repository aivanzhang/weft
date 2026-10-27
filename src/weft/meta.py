import base64
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2

from weft.io import canonical_hash, write_json

MODEL = "muse-spark-1.3"
PRICING = {"input_per_million_usd": 1.25, "output_per_million_usd": 4.25,
           "source": "https://dev.meta.ai/docs/pricing-rate-limits",
           "checked_on": "2026-10-26", "mode": "Standard; no Contributor data-sharing mode"}
INPUT_TOKEN_ENVELOPE = 32768
OUTPUT_TOKEN_LIMIT = 1024
ESTIMATE = INPUT_TOKEN_ENVELOPE * 1.25 / 1e6 + OUTPUT_TOKEN_LIMIT * 4.25 / 1e6


def status():
    available = bool(os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY"))
    return {"credential_present": available, "live_access_verified": False, "model": MODEL,
            "estimated_one_call_envelope_usd": ESTIMATE, "pricing": PRICING,
            "note": "No request made; presence is not account access verification"}


def propose_meta(video, rows, world, out, max_cost_usd):
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        raise ValueError("Meta mode requested, but META_API_KEY / MODEL_API_KEY is absent; no fallback was used")
    if max_cost_usd is None or max_cost_usd < ESTIMATE:
        raise ValueError(f"One bounded call needs an explicit budget of at least ${ESTIMATE:.5f}; estimated, not a provider hard cap")
    public_input = {"world": world.to_dict(), "generation_observations": rows}
    prompt = ("Fit a 1D passive rolling-body model to the attached generation evidence. "
              "Coordinates are image-width units; time is seconds. x(t)=x0+v0*t-sign(v0)*drag*t*t/2 "
              "until velocity reaches zero. No external forces are observed. "
              "Return ONLY a JSON object with numeric v0 in [-4,4], drag in [0,3], and a short reason. "
              "State uncertainty; do not claim measured mass, metric scale, or real intervention validity. "
              "You cannot change any other property, labels, task, or evaluator.\n" + json.dumps(public_input))
    if len(prompt) > 16000:
        raise ValueError("Meta input envelope exceeded")
    content = [{"type": "text", "text": prompt}]
    cap = cv2.VideoCapture(str(video))
    try:
        for row in (rows[0], rows[len(rows) // 2], rows[-1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Could not decode a generation frame for Meta")
            frame = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise ValueError("Could not encode Meta input frame")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buffer).decode(), "detail": "low"}})
    finally:
        cap.release()
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_object"}, "max_completion_tokens": OUTPUT_TOKEN_LIMIT}
    request = Request("https://api.meta.ai/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    started = time.perf_counter()
    record = {"provider": "Meta Model API", "model": MODEL, "request_sha256": canonical_hash(payload),
              "prompt": prompt, "frame_ids": [rows[i]["id"] for i in (0, len(rows) // 2, len(rows) - 1)],
              "estimated_cost_usd": ESTIMATE, "budget_usd": max_cost_usd, "pricing": PRICING,
              "retries": 0, "request_count": 1}
    try:
        with urlopen(request, timeout=45) as response:
            body = response.read(131073)
        if len(body) > 131072:
            raise ValueError("Meta response exceeds 128 KiB")
        result = json.loads(body)
        output = json.loads(result["choices"][0]["message"]["content"])
        if set(output) != {"v0", "drag", "reason"} or not isinstance(output["reason"], str):
            raise ValueError("Meta returned an invalid proposal schema")
        patch = {"v0": output["v0"], "drag": output["drag"]}
        world.patch(patch)
        usage = result.get("usage")
        cost = None if not usage else (usage.get("prompt_tokens", 0) * 1.25 + usage.get("completion_tokens", 0) * 4.25) / 1e6
        record.update(status="completed", response=output, usage=usage, cost_usd=cost)
        return {"method": MODEL, "patch": patch, "reason": output["reason"][:2000]}
    except HTTPError as error:
        record.update(status="failed", error=f"HTTP {error.code}; no automatic retry")
        raise RuntimeError(record["error"]) from None
    except Exception as error:
        record.update(status="failed", error=type(error).__name__)
        raise
    finally:
        record["seconds"] = time.perf_counter() - started
        write_json(out / "meta-call.json", record)
