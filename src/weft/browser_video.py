"""MP4 writing a browser can actually play.

OpenCV's usual `mp4v` fourcc is MPEG-4 Part 2, which Chrome, Safari and Firefox all
refuse: the player renders black and reports an unsupported source. Anything a person
may watch in a page has to be H.264.
"""


def open_writer(cv2, path, fps, size):
    """H.264 where the build supports it, falling back only if it does not."""
    for tag in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise ValueError("No usable MP4 encoder is available")


def playable_in_browser(path):
    """True when the file carries an H.264 sample entry."""
    try:
        with open(path, "rb") as handle:
            return b"avc1" in handle.read(64 << 20)
    except OSError:
        return False
