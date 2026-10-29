"""Synthetic wire-protocol controls, never inferred quality of decoded masks."""

import pytest

from weft.meta_segmentation_parser import PARSER_CONTRACT, SamProtocolError, SamStream


LANE = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
PAYLOAD = "!!!!!(QO(0lu8?"


def record(frame=0, *, object_id="1", payload=PAYLOAD, confidence=".75", mask_confidence=".25", reverse=False):
    box_c = f";c={confidence}" if confidence is not None else ""
    mask_c = f";c={mask_confidence}" if mask_confidence is not None else ""
    box = f"<|box;x1=10;y1=20;x2=14;y2=24;w=1920;h=1080{box_c}|>"
    mask = f"<|mask;x=0;y=0{mask_c};data=5,5,{payload}|>"
    return f"<{frame}f>{object_id}" + (mask + box if reverse else box + mask)


def delta(text, **lane):
    return {"type": "response.output_text.delta", **LANE, **lane, "delta": text}


def final(text, *, part=False, **lane):
    return {"type": "response.content_part.done" if part else "response.output_text.done", **LANE, **lane,
            **({"part": {"type": "output_text", "text": text}} if part else {"text": text})}


def completed():
    return {"type": "response.completed", "response": {"id": "resp_123", "usage": {"video_frames_processed": 354},
                                                       "output": [{"bogus": "aggregate is deliberately unused"}]}}


def parse(text):
    stream = SamStream()
    stream.feed(final(text))
    stream.feed(completed())
    return stream.finish()


def test_full_slot_mapping_halfopen_boxes_and_opaque_mask_confidences():
    result = parse(record(353) + "\n" + record(0, reverse=True))
    assert len(result["frames"]) == 354
    assert result["output_frame_order"] == [353, 0]
    assert result["returned_frames"] == 2 and result["missing_frames"] == 352
    slot = result["frames"][353]
    assert slot["box_xyxy_halfopen"] == [10, 20, 15, 25]
    assert slot["box_xywh"] == [10, 20, 5, 5]
    assert slot["confidence"] == slot["box_confidence"] == .75
    assert slot["mask_confidence"] == .25
    assert slot["mask_encoded"] == PAYLOAD
    assert slot["mask_dimensions"] == [5, 5]
    assert slot["mask_status"] == "opaque_not_decoded_or_codec_validated"
    assert not any(name in slot for name in ("mask", "raster", "mask_centroid", "area"))
    assert result["frames"][1]["status"] == "missing"
    assert result["frames"][1]["box_xyxy_halfopen"] is None
    assert result["frames"][1]["confidence"] is None
    assert result["response_id"] == "resp_123"
    assert result["usage"] == {"video_frames_processed": 354}


def test_every_character_chunk_boundary_and_both_finalizers_exactly_once():
    text = record(7) + "\r\n" + record(12) + "\n"
    stream = SamStream()
    for char in text:
        stream.feed(delta(char))
    stream.feed(final(text, part=True))
    stream.feed(final(text))
    stream.feed(completed())
    result = stream.finish()
    assert result["raw_text"] == text
    assert result["returned_frames"] == 2
    assert result["text_bytes"] == len(text)
    assert len(result["finalizers"]) == 2


def test_payload_equal_sign_and_confidence_lookalike_are_never_reparsed():
    payload = "!abcdec=.75[]()$*+^`{:?"
    result = parse(record(payload=payload, confidence="1e-05", mask_confidence="-0"))
    row = result["frames"][0]
    assert row["mask_encoded"] == payload
    assert row["box_confidence"] == 1e-5 and row["mask_confidence"] == 0
    assert parse(record(payload="~abcde!()[{c=9", confidence=None, mask_confidence=None))["frames"][0]["mask_encoding"] == "lossless"


def test_empty_explicitly_finalized_lane_is_complete_missing_not_failure():
    result = parse("")
    assert result["returned_frames"] == 0 and result["missing_frames"] == 354
    assert all(row["status"] == "missing" for row in result["frames"])
    assert result["raw_text"] == ""


@pytest.mark.parametrize("text", [
    "<0f>", "hello", " \n", record(354), record(-1), record(object_id="0"), record(object_id="01"),
    record() + "\n" + record(), record()+",1"+record().split("1",1)[1],
    record().replace("<0f>", "<0f;extra=1>"),
    record().replace(";x1=10", ";extra=value;x1=10"),
    record().replace(";x1=10", ";x1=10;x1=10"),
    record().replace(";c=.75", ";c=.75;c=.75"),
    record().replace(";x1=10", ";x1"),
    record().replace("<|box", "<|point"),
    record().replace(";x1=10", ";x1=-1"),
    record().replace(";x2=14", ";x2=9"),
    record().replace(";x2=14", ";x2=1920"),
    record().replace(";w=1920", ";w=1919"),
    record().replace(";x1=10", ";x1=10.0"),
    record().replace(";x1=10", ";x1=１0"),
    record().replace("mask;x=0", "mask;x=1"),
    record().replace("data=5,5,", "data=4097,5,"),
    record().replace("data=5,5,", "data=4096,4096,"),
    record().replace("data=5,5,", "data=0,5,"),
    record(payload="base64=="), record(payload="!abcd"), record(payload="!abcde~"),
    record(payload="!abcde}"), record(payload="!abcde|"), record(confidence="NaN"),
    record(confidence="1.01"), record(confidence="1e999"), record(confidence="-1"),
    record(mask_confidence="none"), record()+"trailing", record().replace("<|mask", "extra<|mask"),
])
def test_malformed_nonempty_outputs_fail_closed_without_dropping_content(text):
    with pytest.raises(SamProtocolError):
        parse(text)


@pytest.mark.parametrize("events", [
    [delta(record()), completed()],
    [completed()],
    [final(record()), final(record())],
    [delta(record()), final(record(1))],
    [final(record()), final(record(1), part=True)],
    [final(record()), delta("")],
    [delta(record()), final(record(), content_index=1)],
    [delta(record()), delta("", item_id="msg_2")],
    [delta(record(), output_index=True)],
    [delta(record(), content_index=-1)],
    [delta(record(), item_id="")],
    [final(record()), completed(), completed()],
    [{"type": "response.refusal.delta", "delta": "no"}],
    [{"type": "response.refusal.done", "refusal": "no"}],
    [{"type": "response.content_part.done", "part": {"type": "refusal", "refusal": "no"}}],
    [{"type": "response.failed", "response": {"id": "resp_123"}}],
    [final(record()), {"type": "response.incomplete", "response": {"id": "resp_123"}}],
    [{"type": "unknown.future.event"}], [{"type": []}],
])
def test_protocol_failures_are_explicit_and_sticky(events):
    stream = SamStream()
    with pytest.raises(SamProtocolError):
        for event in events:
            stream.feed(event)
        stream.finish()
    with pytest.raises(SamProtocolError):
        stream.finish()


def test_eof_and_aggregate_output_never_finalize_a_lane():
    stream = SamStream()
    stream.feed({"type": "response.output_item.done", "item": {"content": [{"type": "output_text", "text": record()}]}})
    stream.feed({"type": "response.content_part.done", "part": {"type": "reasoning_text", "text": record()}})
    with pytest.raises(SamProtocolError, match="finalized"):
        stream.feed(completed())
    stream = SamStream()
    stream.feed(final(record()))
    with pytest.raises(SamProtocolError, match="response.completed"):
        stream.finish()


def test_completion_usage_survives_missing_finalizer_for_settlement():
    stream = SamStream()
    with pytest.raises(SamProtocolError):
        stream.feed(completed())
    assert stream.response_id == "resp_123"
    assert stream.usage == {"video_frames_processed": 354}


def test_result_and_provider_usage_mutation_cannot_modify_collector():
    stream = SamStream()
    text = record()
    terminal = completed()
    stream.feed(final(text))
    stream.feed(terminal)
    terminal["response"]["usage"]["video_frames_processed"] = 0
    result = stream.finish()
    result["frames"][0]["box_xywh"][0] = 999
    result["usage"]["video_frames_processed"] = 1
    again = stream.finish()
    assert again["frames"][0]["box_xywh"][0] == 10
    assert again["usage"]["video_frames_processed"] == 354


def test_each_distinct_finalizer_may_occur_once_in_either_order():
    for first, second in ((False, True), (True, False)):
        stream = SamStream()
        stream.feed(final(record(), part=first))
        stream.feed(final(record(), part=second))
        stream.feed(completed())
        assert stream.finish()["returned_frames"] == 1
        repeated = SamStream()
        repeated.feed(final(record(), part=first))
        repeated.feed(final(record(), part=second))
        with pytest.raises(SamProtocolError, match="Repeated"):
            repeated.feed(final(record(), part=first))


def test_complete_354_frame_output_stays_bounded_and_preserves_order():
    text = "\n".join(record(index) for index in reversed(range(354)))
    stream = SamStream()
    for start in range(0, len(text), 1024):
        stream.feed(delta(text[start:start+1024]))
    stream.feed(final(text))
    stream.feed(completed())
    result = stream.finish()
    assert result["returned_frames"] == 354
    assert result["output_frame_order"] == list(reversed(range(354)))
    assert [row["frame"] for row in result["frames"]] == list(range(354))
    assert result["text_bytes"] == len(text)


@pytest.mark.parametrize("limit,value,event", [
    ("maximum_events", 1, delta("")),
    ("maximum_text_bytes", 5, delta("abcdef")),
    ("maximum_line_bytes", 5, delta("abcdef")),
    ("maximum_mask_characters", 6, final(record())),
])
def test_resource_limit_exhaustion_fails_closed(monkeypatch, limit, value, event):
    monkeypatch.setitem(PARSER_CONTRACT, limit, value)
    stream = SamStream()
    if limit == "maximum_events":
        stream.feed({"type": "response.in_progress"})
    with pytest.raises(SamProtocolError, match="bound|exceeds"):
        stream.feed(event)


def test_blank_line_flood_has_a_frame_relative_bound():
    stream = SamStream(frame_count=1)
    with pytest.raises(SamProtocolError, match="line count"):
        stream.feed(delta("\n" * 4))


def test_optional_comma_and_field_order_do_not_change_coordinates():
    text = record().replace("<0f>1", "<0f>,1").replace("x1=10;y1=20", " y1 = 20 ; x1 = 10 ")
    assert parse(text)["frames"][0]["box_xywh"] == [10, 20, 5, 5]


def test_raster_resolution_is_independent_of_source_box_placement():
    row = parse(record().replace("data=5,5,", "data=4,8,"))["frames"][0]
    assert row["mask_dimensions"] == [4, 8]
    assert row["box_xyxy_halfopen"] == [10, 20, 15, 25]
    assert row["mask_encoded"] == PAYLOAD
    assert row["mask_status"] == "opaque_not_decoded_or_codec_validated"


@pytest.mark.parametrize("settings", [{"frame_count": True}, {"frame_count": 15001}, {"width": 0},
                                       {"height": 5000}, {"width": 4096, "height": 4096}])
def test_source_envelope_is_validated_before_streaming(settings):
    with pytest.raises(SamProtocolError):
        SamStream(**settings)


def multi_parse(text, object_ids=(1, 2)):
    stream = SamStream(frame_count=4, object_ids=object_ids)
    for start in range(0, len(text), 7):
        stream.feed(delta(text[start:start + 7]))
    stream.feed(final(text))
    stream.feed(completed())
    return stream.finish()


def test_multi_object_frame_slots_preserve_ids_order_missing_and_confidences():
    second = record(2, object_id='2', confidence='.4', reverse=True)
    text = second + ',' + record(2)[len('<2f>'):] + '\n' + record(0, object_id='2')
    result = multi_parse(text)
    assert result['object_ids'] == ['1', '2']
    assert result['returned_frames'] == 2 and result['missing_frames'] == 2
    assert result['returned_objects'] == 3 and result['missing_objects'] == 5
    assert [row['status'] for row in result['frames']] == ['partial', 'missing', 'returned', 'missing']
    row = result['frames'][2]
    assert 'box_xywh' not in row
    assert [obj['object_id'] for obj in row['objects']] == ['1', '2']
    assert [obj['confidence'] for obj in row['objects']] == [.75, .4]
    assert row['objects'][1]['output_record_index'] == 0
    assert row['objects'][0]['output_record_index'] == 1
    assert result['frames'][0]['objects'][0]['box_xywh'] is None


@pytest.mark.parametrize('text', [
    record() + ',' + record()[len('<0f>'):],
    record() + ',' + record(object_id='3')[len('<0f>'):],
    record() + ',' + record(object_id='2')[len('<0f>'):] + ',',
    record() + ',,' + record(object_id='2')[len('<0f>'):],
    record() + record(object_id='2')[len('<0f>'):],
    record() + '\n' + record(object_id='2'),
])
def test_multi_object_conflicts_and_malformed_segments_fail_whole_stream(text):
    with pytest.raises(SamProtocolError):
        multi_parse(text)


@pytest.mark.parametrize('ids', [(), (1, 1), (0,), (9,), (True,), ('1',), list(range(1, 10)), {1}])
def test_prompted_identity_configuration_is_strict(ids):
    with pytest.raises(SamProtocolError):
        SamStream(object_ids=ids)


def test_eight_objects_and_single_nondefault_id_keep_compatible_fields():
    text = '<0f>' + ','.join(record(object_id=str(i))[len('<0f>'):] for i in range(8, 0, -1))
    result = multi_parse(text, tuple(range(1, 9)))
    assert result['returned_objects'] == 8 and result['missing_objects'] == 24
    single = multi_parse(record(object_id='8'), (8,))['frames'][0]
    assert single['object_id'] == '8' and single['objects'][0]['object_id'] == '8'
    single['box_xywh'][0] = 999
    assert single['objects'][0]['box_xywh'][0] == 10
