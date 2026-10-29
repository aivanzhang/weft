"""Bounded SAM text/SSE validation; encoded masks stay opaque, never decoded."""

from copy import deepcopy
import math
import re


PARSER_CONTRACT = {
    "method": "sam-multi-object-opaque-mask-stream-v2",
    "maximum_objects": 8,
    "maximum_text_bytes": 8_000_000,
    "maximum_events": 50_000,
    "maximum_line_bytes": 256_000,
    "maximum_mask_characters": 250_000,
    "maximum_frames": 15_000,
    "maximum_source_dimension": 4096,
    "maximum_mask_area": 8_847_360,
    "unknown_text_fields": "rejected_not_silently_discarded",
    "mask_validation": "wire_syntax_and_dimensions_only_not_codec_or_raster",
    "protocol_source": "https://github.com/meta-models/meta-sam/blob/main/protocol/sam3.md",
    "alphabet_conflict_resolution": "Protocol forbids interior tilde; API overview says it may occur. Follow the explicit protocol alphabet.",
    "mask_dimensions": "Independently declared raster dimensions; box defines placement, not raster resolution",
}
BASE85_DIGITS = frozenset("!#$%&'()*+-./0123456789:=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{")
_UNSIGNED = re.compile(r"[0-9]+\Z", re.ASCII)
_SIGNED = re.compile(r"-?[0-9]+\Z", re.ASCII)
_CONFIDENCE = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z", re.ASCII)
_FRAME = re.compile(r"<([0-9]+)f>", re.ASCII)
_OBJECT = re.compile(r"[0-9]+", re.ASCII)
_IGNORED_EVENTS = frozenset({
    "response.created", "response.queued", "response.in_progress", "response.output_item.added",
    "response.output_item.done", "response.content_part.added", "response.reasoning_text.delta",
    "response.reasoning_text.done", "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
})


class SamProtocolError(ValueError):
    """The stream cannot establish a complete structurally valid segmentation result."""


class SamStream:
    """One bounded multi-object video stream; feed decoded SSE dictionaries, then finish at EOF."""

    def __init__(self, frame_count=354, width=1920, height=1080, object_ids=(1,)):
        for value, limit, name in ((frame_count, PARSER_CONTRACT["maximum_frames"], "frame count"),
                                   (width, PARSER_CONTRACT["maximum_source_dimension"], "width"),
                                   (height, PARSER_CONTRACT["maximum_source_dimension"], "height")):
            if type(value) is not int or not 1 <= value <= limit:
                raise SamProtocolError(f"Invalid bounded source {name}")
        if width * height > PARSER_CONTRACT["maximum_mask_area"]:
            raise SamProtocolError("Source area exceeds the parser envelope")
        if (not isinstance(object_ids, (list, tuple)) or not 1 <= len(object_ids) <= 8
                or any(type(value) is not int or not 1 <= value <= 8 for value in object_ids)
                or len(set(object_ids)) != len(object_ids)):
            raise SamProtocolError("Expected 1..8 unique integer object IDs within 1..8")
        self.object_ids = tuple(str(value) for value in object_ids)
        self.frame_count, self.width, self.height = frame_count, width, height
        self.response_id, self.usage = None, None
        self._lane = None
        self._chunks, self._buffer = [], ""
        self._text_bytes, self._events = 0, 0
        self._deltas_seen, self._completed = False, False
        self._finalizers = set()
        self._final_text = None
        self._error = None
        self._records, self._frame_order = {}, []
        self._ignored = {}
        self._line_number = 0
        self._record_count = 0

    def _fail(self, message):
        self._error = message
        raise SamProtocolError(message)

    def _identity(self, event):
        item = event.get("item_id")
        indices = (event.get("output_index"), event.get("content_index"))
        if (not isinstance(item, str) or not 1 <= len(item) <= 256
                or any(type(value) is not int or not 0 <= value <= 1024 for value in indices)):
            self._fail("Output text requires a bounded item_id/output_index/content_index lane")
        identity = (item, *indices)
        if self._lane is not None and identity != self._lane:
            self._fail("Interleaved or conflicting output-text lanes")
        self._lane = identity

    def _metadata(self, event):
        response = event.get("response")
        if response is None:
            return
        if not isinstance(response, dict):
            self._fail("Response metadata must be an object")
        identifier = response.get("id")
        if identifier is not None:
            if not isinstance(identifier, str) or not 1 <= len(identifier) <= 256:
                self._fail("Invalid response ID")
            if self.response_id is not None and identifier != self.response_id:
                self._fail("Response ID changed within one stream")
            self.response_id = identifier
        if "usage" in response:
            if response["usage"] is not None and not isinstance(response["usage"], dict):
                self._fail("Usage metadata must remain an object or null")
            self.usage = deepcopy(response["usage"])

    def feed(self, event):
        if not isinstance(event, dict):
            self._fail("SSE event must be a decoded object")
        if self._error is not None:
            raise SamProtocolError(self._error)
        self._events += 1
        if self._events > PARSER_CONTRACT["maximum_events"]:
            self._fail("SSE event count exceeds the parser bound")
        if self._completed:
            self._fail("An event followed response.completed")
        self._metadata(event)
        kind = event.get("type")
        if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
            self._fail("SSE event type must be bounded text")
        if kind in {"error", "response.failed", "response.error", "response.refusal.delta", "response.refusal.done"}:
            self._fail(f"Terminal provider failure: {kind}")
        if kind == "response.incomplete":
            self._fail("Explicit incomplete response is not a completed segmentation")
        if kind == "response.completed":
            if not isinstance(event.get("response"), dict):
                self._fail("Completed event requires response metadata")
            if not self._finalizers:
                self._fail("Completed response lacks an explicitly finalized output-text lane")
            self._completed = True
            return
        if kind in _IGNORED_EVENTS:
            self._ignored[kind] = self._ignored.get(kind, 0) + 1
            return
        if kind == "response.content_part.done":
            part = event.get("part")
            if not isinstance(part, dict):
                self._fail("Content-part finalizer requires a part object")
            if part.get("type") == "refusal":
                self._fail("Terminal provider refusal part")
            if part.get("type") == "reasoning_text":
                self._ignored[kind + ":reasoning_text"] = self._ignored.get(kind + ":reasoning_text", 0) + 1
                return
            if part.get("type") != "output_text":
                self._fail("Unknown finalized content part")
            self._identity(event)
            self._finalize(kind, part.get("text"))
            return
        if kind not in {"response.output_text.delta", "response.output_text.done"}:
            self._fail("Unknown SSE event type")
        self._identity(event)
        if kind == "response.output_text.done":
            self._finalize(kind, event.get("text"))
            return
        if self._finalizers:
            self._fail("Output text delta followed finalization")
        delta = event.get("delta")
        if not isinstance(delta, str):
            self._fail("Output-text delta must be a string")
        self._deltas_seen = True
        self._push(delta)

    def _finalize(self, kind, value):
        if not isinstance(value, str):
            self._fail("Output-text finalizer must contain a string")
        if len(value.encode("utf-8")) > PARSER_CONTRACT["maximum_text_bytes"]:
            self._fail("Finalized text exceeds the parser byte bound")
        if kind in self._finalizers:
            self._fail("Repeated finalizer event type")
        if self._finalizers:
            if value != self._final_text:
                self._fail("Distinct finalizers disagree")
            self._finalizers.add(kind)
            return
        if self._deltas_seen and value != "".join(self._chunks):
            self._fail("Finalized text differs from accumulated deltas")
        if not self._deltas_seen:
            self._push(value)
        if self._buffer:
            self._line(self._buffer)
            self._buffer = ""
        if value and not self._records:
            self._fail("Nonempty segmentation text contains no records")
        self._final_text = value
        self._finalizers.add(kind)

    def _push(self, chunk):
        length = len(chunk.encode("utf-8"))
        if self._text_bytes + length > PARSER_CONTRACT["maximum_text_bytes"]:
            self._fail("Output text exceeds the parser byte bound")
        if not chunk.isascii():
            self._fail("Segmentation wire text must be ASCII")
        self._text_bytes += length
        self._chunks.append(chunk)
        buffered = self._buffer + chunk
        start = 0
        while (end := buffered.find("\n", start)) >= 0:
            self._line(buffered[start:end])
            start = end + 1
        self._buffer = buffered[start:]
        if len(self._buffer.encode("utf-8")) > PARSER_CONTRACT["maximum_line_bytes"]:
            self._fail("Buffered line exceeds the parser byte bound")

    def _fields(self, body, allowed, required):
        fields = {}
        for part in body.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                self._fail("Token field is missing its value")
            key, value = (piece.strip() for piece in part.split("=", 1))
            if key not in allowed:
                self._fail(f"Unknown token field: {key[:80]}")
            if key in fields:
                self._fail(f"Duplicate token field: {key}")
            fields[key] = value
        if not required <= fields.keys():
            self._fail("Token is missing required fields")
        return fields

    def _integer(self, value, *, signed=False):
        if len(value) > 16 or not (_SIGNED if signed else _UNSIGNED).fullmatch(value):
            self._fail("Invalid bounded ASCII integer")
        return int(value)

    def _confidence(self, fields):
        if "c" not in fields:
            return None
        value = fields["c"]
        if len(value) > 64 or not _CONFIDENCE.fullmatch(value):
            self._fail("Malformed confidence")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            self._fail("Confidence must be finite and within [0, 1]")
        return 0. if number == 0 else number

    def _line(self, line):
        self._line_number += 1
        if self._line_number > self.frame_count * 2 + 1:
            self._fail("Segmentation line count exceeds the bounded frame envelope")
        if len(line.encode("utf-8")) > PARSER_CONTRACT["maximum_line_bytes"]:
            self._fail("Segmentation line exceeds the parser byte bound")
        line = line.strip()
        if not line:
            return
        marker = _FRAME.match(line)
        if marker is None:
            self._fail("Malformed explicit frame marker or unknown frame fields")
        frame = self._integer(marker.group(1))
        if frame >= self.frame_count:
            self._fail("Frame index exceeds the pinned source")
        if frame in self._records:
            self._fail("Duplicate frame/object record")
        offset = marker.end()
        records = {}
        while offset < len(line):
            if line[offset] == ",":
                offset += 1
            identity = _OBJECT.match(line, offset)
            if identity is None or identity.group() not in self.object_ids:
                self._fail("Expected an explicitly prompted object ID")
            object_id = identity.group()
            if object_id in records:
                self._fail("Duplicate frame/object record")
            offset = identity.end()
            tokens = {}
            while line.startswith("<|", offset):
                end = line.find("|>", offset + 2)
                if end < 0:
                    self._fail("Unclosed segmentation token")
                name, separator, body = line[offset+2:end].partition(";")
                if name not in {"box", "mask"} or not separator:
                    self._fail("Unknown or malformed segmentation token")
                if name in tokens:
                    self._fail("Repeated box or mask token")
                tokens[name] = body
                offset = end + 2
            if set(tokens) != {"box", "mask"}:
                self._fail("Each object requires exactly one box and one mask")
            records[object_id] = self._record(frame, object_id, tokens, len(records))
            if offset < len(line) and line[offset] != ",":
                self._fail("Object segments require a comma separator")
        if not records:
            self._fail("Frame marker lacks object records")
        self._records[frame] = records
        self._record_count += len(records)
        self._frame_order.append(frame)

    def _record(self, frame, object_id, tokens, index):
        box_fields = self._fields(tokens["box"], {"x1", "y1", "x2", "y2", "w", "h", "c"}, {"x1", "y1", "x2", "y2", "w", "h"})
        mask_fields = self._fields(tokens["mask"], {"x", "y", "data", "c"}, {"x", "y", "data"})
        x1, y1, x2, y2 = (self._integer(box_fields[key], signed=True) for key in ("x1", "y1", "x2", "y2"))
        width, height = (self._integer(box_fields[key]) for key in ("w", "h"))
        if (width, height) != (self.width, self.height):
            self._fail("Box source dimensions differ from the pinned source")
        if not 0 <= x1 <= x2 < width or not 0 <= y1 <= y2 < height:
            self._fail("Inclusive source box is reversed or out of bounds")
        if any(self._integer(mask_fields[key]) != 0 for key in ("x", "y")):
            self._fail("Mask origin must be zero relative to its box")
        dimensions = mask_fields["data"].split(",", 2)
        if len(dimensions) != 3:
            self._fail("Mask requires height,width,encoded-payload")
        mask_height, mask_width = (self._integer(value) for value in dimensions[:2])
        encoded = dimensions[2]
        if (not 1 <= mask_width <= PARSER_CONTRACT["maximum_source_dimension"]
                or not 1 <= mask_height <= PARSER_CONTRACT["maximum_source_dimension"]
                or mask_width * mask_height > PARSER_CONTRACT["maximum_mask_area"]):
            self._fail("Mask dimensions exceed the bounded raster envelope")
        if (not 6 <= len(encoded) <= PARSER_CONTRACT["maximum_mask_characters"] or encoded[0] not in "!~"
                or any(value not in BASE85_DIGITS for value in encoded[1:])):
            self._fail("Invalid bounded opaque mask marker or base85 alphabet")
        box_confidence, mask_confidence = self._confidence(box_fields), self._confidence(mask_fields)
        return {"frame": frame, "status": "returned", "object_id": object_id,
                                "box_xyxy_halfopen": [x1, y1, x2+1, y2+1], "box_xywh": [x1, y1, x2-x1+1, y2-y1+1],
                                "confidence": box_confidence, "box_confidence": box_confidence, "mask_confidence": mask_confidence,
                                "mask_encoded": encoded, "mask_dimensions": [mask_height, mask_width],
                                "mask_encoding": "one_bit" if encoded[0] == "!" else "lossless",
                                "mask_status": "opaque_not_decoded_or_codec_validated", "source_dimensions": [width, height],
                                "output_record_index": self._record_count + index, "source_line_number": self._line_number}

    def finish(self):
        if self._error is not None:
            raise SamProtocolError(self._error)
        if not self._completed:
            self._fail("Incomplete stream: response.completed was not received")
        frames = []
        returned_objects = 0
        for index in range(self.frame_count):
            objects = []
            for object_id in self.object_ids:
                row = self._records.get(index, {}).get(object_id)
                if row is None:
                    row = {"frame": index, "status": "missing", "object_id": object_id,
                           "box_xyxy_halfopen": None, "box_xywh": None, "confidence": None,
                           "box_confidence": None, "mask_confidence": None, "mask_encoded": None,
                           "mask_dimensions": None, "mask_encoding": None, "mask_status": "not_returned",
                           "source_dimensions": [self.width, self.height]}
                objects.append(deepcopy(row))
            count = sum(row["status"] == "returned" for row in objects)
            returned_objects += count
            frame = {"frame": index, "status": "returned" if count == len(objects) else "partial" if count else "missing",
                     "objects": objects}
            if len(objects) == 1:
                frame.update(deepcopy(objects[0]))
            frames.append(frame)
        return {"schema_version": 2, "object_ids": list(self.object_ids), "contract": deepcopy(PARSER_CONTRACT), "frames": frames,
                "response_id": self.response_id, "usage": deepcopy(self.usage), "status": "completed",
                "lane": {"item_id": self._lane[0], "output_index": self._lane[1], "content_index": self._lane[2]},
                "raw_text": self._final_text, "output_frame_order": self._frame_order.copy(),
                "returned_objects": returned_objects, "missing_objects": self.frame_count * len(self.object_ids) - returned_objects,
                "returned_frames": len(self._records), "missing_frames": self.frame_count-len(self._records),
                "events_received": self._events, "text_bytes": self._text_bytes,
                "finalizers": sorted(self._finalizers), "ignored_event_types": deepcopy(self._ignored),
                "scope": "Wire-valid source boxes and opaque mask payloads; no raster, centroid, mask-quality or physical-identity claim"}
