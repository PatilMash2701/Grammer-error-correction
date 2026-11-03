"""End-to-end test of all 3 models via API."""
import urllib.request
import urllib.error
import json

BASE = "http://localhost:9000"
TEST = "She go to school yesterday and dont came back."

def correct(text, model):
    payload = json.dumps({"text": text, "model": model, "num_beams": 2}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/v1/correct",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        r = urllib.request.urlopen(req, timeout=60)
        data = json.loads(r.read())
        return data["corrected_text"], data["processing_time_ms"]
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode()[:100]}", 0
    except Exception as e:
        return f"Error: {e}", 0

print(f"Input:  \"{TEST}\"\n")
for model in ["t5", "coedit", "llama"]:
    corrected, ms = correct(TEST, model)
    print(f"[{model:6s}] {corrected!r}  ({ms:.0f}ms)")
