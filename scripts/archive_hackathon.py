#!/usr/bin/env python3
"""Archive the event's first-hop public references without submitting forms."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hackathon/sources"
RAW = ROOT / ".local/source-html"
BASE = "https://cerebralvalley.ai/e/meta-ai-developer-hackathon"
TARGETS = [
    ("event", BASE),
    ("official-rules", BASE + "/waiver"),
    ("discord", BASE + "/discord"),
    ("livestream", BASE + "/live"),
    ("meta-event", "https://dev.meta.ai/events/global-hackathon"),
    *[("meta-" + slug, "https://dev.meta.ai/models/" + slug) for slug in ["muse-spark", "muse-wonder", "muse-image", "muse-voice-transcribe", "sam-3-1", "muse-glimmer", "muse-spark-open"]],
    ("meta-model-api", "https://dev.meta.ai/products/meta-model-api"),
    ("meta-muse-code", "https://dev.meta.ai/lp/muse-code"),
    *[("meta-docs-" + slug, "https://dev.meta.ai/docs/" + slug) for slug in ["overview", "quickstart", "models"]],
    ("meta-api-cookbook", "https://github.com/meta-models/meta-model-cookbook"),
    ("meta-api-fundamentals", "https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals"),
    ("meta-code-recipes", "https://github.com/meta-models/meta-model-cookbook/tree/main/04_muse_code"),
    ("meta-oss-cookbook", "https://github.com/meta-models/meta-oss-cookbook"),
    *[("meta-video-" + slug, "https://developers.meta.com/resources/videos/" + slug + "/") for slug in ["multimodal-intelligence-of-muse-spark", "building-with-muse-spark-and-muse-code", "the-road-ahead-superintelligence-blueprint"]],
    ("baseten", "https://www.baseten.co/"),
    *[("baseten-" + slug.rsplit("/", 1)[-1], "https://docs.baseten.co/" + slug) for slug in ["quickstart", "reference/inference-api/chat-completions", "inference/model-apis/reasoning", "inference/model-apis/vision", "inference/model-apis/audio"]],
    ("modal", "https://modal.com/"),
    *[("modal-" + slug, "https://modal.com/docs/guide/" + slug) for slug in ["intro", "endpoints", "gpu", "sandboxes", "training-quickstart"]],
    ("fireworks", "https://fireworks.ai/"),
    *[("credits-" + name, "https://docs.google.com/forms/d/e/" + ident + "/viewform") for name, ident in [
        ("meta", "1FAIpQLScam78z7tHsNYibiFyJQVFzEgr0coyBTiY568SxxNbsPa3jiw"),
        ("baseten", "1FAIpQLSfM7gcK8yMZEmAGq6DbTyiuv2c71IH7CKCy0fXGIfUED1qxVQ"),
        ("modal", "1FAIpQLSe_GDPtLZsdA57xPhFcOCi_yLjPf8HRxMIQJ5V7ystoKSqWCQ"),
        ("fireworks", "1FAIpQLSdPQTTES8_nhavzZkOrebVNYbM1FumWLdpeZx8ZSNe4QLXKPA"),
    ]],
    ("speaker-suraj", "https://www.linkedin.com/in/surajsubramanian/"),
    ("speaker-yash", "https://www.linkedin.com/in/yash-khare-b54a99131"),
    ("speaker-sinan", "https://www.linkedin.com/in/sinan-ozdemir/"),
    ("meta-privacy", "https://www.facebook.com/privacy/policy/"),
    ("meta-api-terms", "https://dev.meta.ai/legal/terms-of-service/"),
    ("meta-api-acceptable-use", "https://dev.meta.ai/legal/acceptable-use-policy/"),
    ("credits-meta-terms-landing", "https://www.facebook.com/business/f/1797166494659033"),
    ("host-meta", "https://cerebralvalley.ai/u/meta"),
    ("host-cerebral-valley", "https://cerebralvalley.ai/u/cv"),
]


def rich_html(node):
    kind = node.get("type", "")
    content = "".join(rich_html(c) for c in node.get("content", []))
    if kind == "text":
        content = escape(node.get("text", ""))
        for mark in node.get("marks", []):
            mt = mark.get("type")
            if mt == "link":
                content = '<a href="' + escape(mark["attrs"]["href"], quote=True) + '">' + content + "</a>"
            elif mt in {"bold", "italic", "code"}:
                tag = {"bold": "strong", "italic": "em", "code": "code"}[mt]
                content = f"<{tag}>{content}</{tag}>"
        return content
    tags = {"paragraph": "p", "bulletList": "ul", "orderedList": "ol", "listItem": "li", "table": "table", "tableRow": "tr", "tableHeader": "th", "tableCell": "td", "blockquote": "blockquote", "codeBlock": "pre"}
    tag = "h" + str(node.get("attrs", {}).get("level", 2)) if kind == "heading" else tags.get(kind)
    if kind == "hardBreak": return "<br>"
    return f"<{tag}>{content}</{tag}>" if tag else content


def event_docs(html):
    docs = []
    for match in re.finditer(r'self\.__next_f\.push\((\[.*?\])\)</script>', html):
        try:
            values = json.loads(match[1])
        except ValueError:
            continue
        for value in values:
            if isinstance(value, str) and value.startswith('{"type":"doc"'):
                try: docs.append(json.loads(value))
                except ValueError: pass
    return docs


class CleanHTML(HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base, self.parts, self.links, self.skip = base, [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style", "svg", "noscript"}:
            self.skip.append(tag)
        if self.skip: return
        if tag == "a" and attrs.get("href"):
            href = urljoin(self.base, attrs["href"])
            self.links.append(href)
            self.parts.append('<a href="' + escape(href, quote=True) + '">')
        elif tag == "img":
            if attrs.get("alt"): self.parts.append(escape(attrs["alt"]))
        else: self.parts.append(f"<{tag}>")

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip[-1]: self.skip.pop()
            return
        self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skip: self.parts.append(escape(data))


def archive(target, previous=None):
    slug, url = target
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    rec = {"id": slug, "url": url, "retrieved_at": stamp, "file": f"sources/{slug}.md"}
    try:
        if previous is not None:
            if not (RAW / f"{slug}.html").exists(): return previous
            body = (RAW / f"{slug}.html").read_bytes()
            stamp = previous["retrieved_at"]
            rec.update({k: previous[k] for k in ["retrieved_at", "final_url", "http_status"]})
        else:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LoomReferenceArchive/1.0)"})
            with urlopen(request, timeout=30) as response:
                body = response.read(8_000_001)
                if len(body) > 8_000_000: raise ValueError("Response exceeds 8 MB archive limit")
                rec.update(final_url=response.url, http_status=response.status)
        (RAW / f"{slug}.html").write_bytes(body)
        rec["sha256"] = sha256(body).hexdigest()
        if body.startswith(b"%PDF-"):
            pdf_text = subprocess.run(["pdftotext", "-layout", str(RAW / f"{slug}.html"), "-"], capture_output=True, text=True, check=True).stdout
            md = "\n\n".join(f"## Source page {i + 1}\n\n" + page.strip() for i, page in enumerate(pdf_text.split("\f")) if page.strip())
            rec.update(status="captured-pdf-text", characters=len(md), links=[], extraction="PDF magic detected; all pages extracted with pdftotext")
            (OUT / f"{slug}.md").write_text(f"# Source snapshot: {slug}\n\n- Source: {url}\n- Retrieved: {stamp}\n- Status: captured-pdf-text\n- Method: read-only GET redirected to a PDF; all pages extracted with pdftotext.\n- SHA-256 of source bytes: `{rec['sha256']}`\n\nOriginal source wording, numbering, and timezone labels retained. No terms accepted.\n\n---\n\n" + md + "\n")
            return rec
        html = body.decode("utf-8", errors="replace")
        if slug in {"event", "official-rules"}:
            docs = event_docs(html)
            needle = "Hackathon Terms & Conditions" if slug == "official-rules" else "Come and build"
            matching = [d for d in docs if needle in json.dumps(d, ensure_ascii=False)]
            if matching:
                html = rich_html(matching[0])
                rec["extraction"] = "Public page rich-text document; compared with rendered browser page"
        elif '<main' in html:
            match = re.search(r'<main\b[^>]*>(.*?)</main>', html, re.S)
            if match: html = match[1]
        cleaner = CleanHTML(rec["final_url"])
        cleaner.feed(html)
        cleaned = "".join(cleaner.parts)
        md = subprocess.run(["pandoc", "--from=html", "--to=gfm-raw_html", "--wrap=none"], input=cleaned, text=True, capture_output=True, check=True).stdout
        md = re.sub(r"\n{3,}", "\n\n", md).strip()
        rec.update(status="captured", characters=len(md), links=sorted(set(cleaner.links)))
        if len(md) < 400 or "Enable JavaScript and cookies to continue" in md or slug == "livestream" or slug.startswith("speaker-"):
            rec["status"] = "limited-content"
        if slug.startswith("meta-video-"):
            rec["status"] = "page-only-no-transcript"
        header = f"# Source snapshot: {slug}\n\n- Source: {url}\n- Retrieved: {stamp}\n- Method: read-only HTTP GET; HTML converted to Markdown.\n- Status: {rec['status']}\n- SHA-256 of source bytes: `{rec['sha256']}`\n\n> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.\n\n---\n\n"
        (OUT / f"{slug}.md").write_text(header + md + "\n")
    except Exception as exc:
        rec.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        (OUT / f"{slug}.md").write_text(f"# Source unavailable: {slug}\n\n- Source: {url}\n- Attempted: {stamp}\n- Result: {rec['error']}\n\nNo page content was captured. Do not treat this file as a source snapshot. No forms were submitted or authentication attempted.\n")
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "hackathon/source-manifest.json"
    existing = {r["id"]: r for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    previous = existing if "--offline" in sys.argv else {}
    targets = [t for t in TARGETS if t[0] in previous] if "--offline" in sys.argv else TARGETS
    if "--missing" in sys.argv: targets = [t for t in TARGETS if t[0] not in existing]
    if existing and not any(flag in sys.argv for flag in ["--offline", "--missing", "--refresh"]):
        raise SystemExit("Archive exists. Use --missing, --offline, or --refresh explicitly.")
    if "--refresh" in sys.argv and existing:
        backup = ROOT / ".local/history" / datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S-%Z")
        import shutil
        shutil.copytree(OUT, backup / "sources")
        shutil.copy2(manifest, backup / "source-manifest.json")
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(lambda target: archive(target, previous.get(target[0])), targets))
    existing.update({r["id"]: r for r in records})
    manifest.write_text(json.dumps(list(existing.values()), indent=2) + "\n")
    print(json.dumps([{k: r.get(k) for k in ["id", "status", "characters", "error"]} for r in records], indent=2))


if __name__ == "__main__": main()
