"""Attributed public video fixtures; downloads are explicit and hash-pinned."""

from pathlib import Path
from urllib.request import Request, urlopen

from weft.io import digest, local_time, write_json


SAMPLES = {
    "single-leg-squat": {
        "file": "single-leg-squat.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/16/Basic_single_leg_squat.webm",
        "source_page": "https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm",
        "author": "RickyBennison", "title": "Basic single leg squat",
        "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "sha256": "ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e",
        "changes": "Original WebM retained unchanged; Weft creates separately labeled derived pose overlays.",
    },
    "squat": {
        "file": "squat.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5c/Squat_-_exercise_demonstration_video.webm",
        "source_page": "https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm",
        "author": "FitnessScape", "title": "Squat - exercise demonstration video",
        "license": "CC BY 3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "sha256": "2440985661c3533a4ce78472b0f4577dbdf023aff3f8f9a225bbb5ff8071b1e9",
        "changes": "Original WebM retained unchanged. Feet are cropped; use as a limited-input example.",
    },
    'kettlebell-pushup': {
        'file': 'kettlebell-pushup.webm',
        'title': 'Kettlebell crush grip push-up',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Kettlebell_crush_grip_push-up.webm',
        'source_page': 'https://commons.wikimedia.org/wiki/File:Kettlebell_crush_grip_push-up.webm',
        'author': 'Taco Fleur / Cavemantraining',
        'license': 'CC BY-SA 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'bytes': 1251790,
        'sha256': '4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc',
        'changes': 'Original WebM retained unchanged; derived pose previews are separate adaptations.',
        'challenge': 'horizontal torso; loaded hands; fixed camera',
    },
    'lunge-twist': {
        'file': 'lunge-twist.webm',
        'title': 'Kettlebell Lunge and Twist',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/2/21/Kettlebell_Lunge_and_Twist.webm',
        'source_page': 'https://commons.wikimedia.org/wiki/File:Kettlebell_Lunge_and_Twist.webm',
        'author': 'Taco Fleur / Cavemantraining',
        'license': 'CC BY-SA 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'bytes': 3005172,
        'sha256': 'afe0c808e05c6b178c90295f9b9deeeda16322a5b6e1b10d8ebeac31c5eb8a34',
        'changes': 'Original WebM retained unchanged; derived pose previews are separate adaptations.',
        'challenge': 'loaded lunge; torso twist; limb occlusion',
    },
    'jumping-jacks-burpees': {
        'author': 'Taco Fleur / Cavemantraining',
        'bytes': 6544177,
        'challenge': 'Lateral limb motion, jumping and floor transitions',
        'changes': 'Original WebM retained unchanged; derived pose previews are separate adaptations.',
        'file': 'jumping-jacks-burpees.webm',
        'license': 'CC BY-SA 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'sha256': '1c8370a29c52b5b1538f03f8c63be6cdc9333dea3cbd77cc8f2af0dbe496d5b4',
        'source_page': 'https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm',
        'title': 'Jumping jacks and burpees',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/5/57/Jumping_jacks_and_burpees.webm',
    },
}


def describe_source(sha256):
    return next((dict(sample) for sample in SAMPLES.values() if sample["sha256"] == sha256), None)


def fetch_exercise_sample(out="data/exercise", name="single-leg-squat"):
    if name not in SAMPLES:
        raise ValueError("Unknown exercise sample")
    root, source = Path(out), SAMPLES[name]
    root.mkdir(parents=True, exist_ok=True)
    path = root / source["file"]
    if not path.exists():
        request = Request(source["url"], headers={"User-Agent": "Weft local research fixture/0.1"})
        with urlopen(request, timeout=30) as response:
            data = response.read(source.get("bytes", 2_000_000) + 1)
        if len(data) > source.get("bytes", 2_000_000):
            raise ValueError("Exercise fixture exceeded its pinned download bound")
        import hashlib
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError("Downloaded exercise video does not match its pinned source")
        with path.open("xb") as stream:
            stream.write(data)
    if digest(path) != source["sha256"]:
        raise ValueError("Existing exercise video differs from its pinned source; preserved without overwrite")
    provenance = path.with_suffix(".source.json")
    if not provenance.exists():
        write_json(provenance, {**source, "verified_at": local_time(), "bytes": path.stat().st_size,
                               "scope": "Recorded human exercise; not measured muscle activation or force"})
    return {"path": str(path), "source": source, "provenance": str(provenance)}
