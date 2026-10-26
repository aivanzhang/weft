#!/usr/bin/env python3
"""Index source artifacts and expose uncaptured first-hop links."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1] / "hackathon"
MANUAL = {
    "participant-guide": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details", "browser-transcription"),
    "submission-form": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit", "browser-fields"),
    "announcements": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon", "browser-transcription"),
    "discord-browser": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon/discord", "browser-transcription"),
    "livestream-browser": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon/live", "page-only-no-transcript"),
    "credit-terms": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon/waiver", "pdf-text-all-3-pages"),
    "application": ("https://cerebralvalley.ai/e/meta-ai-developer-hackathon", "user-provided-browser-verified"),
}


def normalized(url):
    p = urlsplit(url)
    # Tracking parameters and anchors do not identify separate source documents.
    return urlunsplit((p.scheme, p.netloc.removeprefix("www."), p.path.rstrip("/"), "", ""))


def main():
    manifest = ROOT / "source-manifest.json"
    records = {r["id"]: r for r in json.loads(manifest.read_text())}
    for slug, (url, status) in MANUAL.items():
        path = ROOT / "sources" / (slug + ".md")
        if not path.exists(): continue
        text = path.read_text()
        match = re.search(r"^- (?:Observed|Retrieved|Converted): (.+)$", text, re.M)
        records[slug] = {"id": slug, "url": url, "status": status, "file": f"sources/{slug}.md", "retrieved_at": match[1] if match else "See source header", "artifact_sha256": sha256(path.read_bytes()).hexdigest(), "characters": len(text)}
        if slug == "credit-terms":
            pdf = re.search(r"^- Source PDF: (.+)$", text, re.M)
            if pdf: records[slug]["url"] = pdf[1]
    manifest.write_text(json.dumps(list(records.values()), indent=2) + "\n")
    counts = Counter(r["status"] for r in records.values())
    lines = ["# Source index", "", "Generated from `source-manifest.json`. Source snapshots retain source ownership. A captured page is not a guarantee that hidden UI, video speech, or private content was captured.", "", f"Total records: **{len(records)}**.", "", "| Source | Capture status | Reference |", "| --- | --- | --- |"]
    for r in sorted(records.values(), key=lambda r:r["id"]):
        assert (ROOT / r["file"]).exists(), r["file"]
        lines.append(f"| [{r['id']}]({r['file']}) | {r['status']} | [Original]({r['url']}) |")
    lines += ["", "## Status counts", ""] + [f"- {status}: {n}" for status,n in sorted(counts.items())]
    (ROOT / "SOURCE_INDEX.md").write_text("\n".join(lines) + "\n")
    by_url = {}
    for r in records.values(): by_url.setdefault(normalized(r["url"]), []).append(r)
    primary = ["event", "participant-guide", "official-rules", "submission-form", "announcements", "discord-browser", "livestream-browser", "credits-meta"]
    found = {}
    for slug in primary:
        text = (ROOT / "sources" / (slug + ".md")).read_text()
        for url in re.findall(r"https?://[^\s<>\])]+", text):
            url = url.rstrip(".,;")
            found.setdefault(normalized(url), {"url":url,"sources":set()})["sources"].add(slug)
    lines = ["# First-hop link inventory", "", "Covers links exposed in the primary event documents, guide, official rules, announcements, submission fields, Discord/livestream access pages, and Meta credit form. Does not recursively crawl unrelated website navigation. Query/fragment variants are deduplicated; original links remain in snapshots.", "", "| Destination | Referenced by | Archived coverage |", "| --- | --- | --- |"]
    gaps = []
    for key,item in sorted(found.items()):
        matches=by_url.get(key, [])
        coverage="; ".join(f"[{r['id']}]({r['file']}) ({r['status']})" for r in matches)
        if not coverage:
            host=urlsplit(item['url']).netloc
            if 'discord.com' in host: coverage="Server destination recorded; private channels not archived; no connection or role changes"
            elif 'vimeo.com' in host: coverage="Player/recording destination recorded; video transcript not captured"
            elif host in {'accounts.google.com','www.google.com','docs.google.com'}: coverage="Ancillary sign-in/help/redirect; not followed"
            elif key == normalized('https://dev.meta.ai'): coverage="Account/dashboard destination; not opened or modified"
            else: coverage="Link recorded; destination not independently captured"; gaps.append(item['url'])
        lines.append(f"| [Source]({item['url']}) | {', '.join(sorted(item['sources']))} | {coverage} |")
    (ROOT / "LINKS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"records":len(records),"status_counts":counts,"first_hop_links":len(found),"uncaptured_other":gaps},indent=2))


if __name__ == "__main__": main()
