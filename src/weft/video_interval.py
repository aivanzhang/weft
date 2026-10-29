"""Hash-pinned decoder intervals; ordinal coverage is not independent encoded-frame completeness."""

from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np


MAX_SOURCE_BYTES = 100_000_000
MAX_SOURCE_FRAMES = 216_000
MAX_SOURCE_SECONDS = 3600
MAX_PREFIX_FRAMES = 20_000
MAX_SELECTED_FRAMES = 1800
MAX_SELECTED_SECONDS = 60
MAX_DEADLINE_SECONDS = 120


class IntervalVideo:
    """Read an inclusive source interval once; use as a context manager and consume every frame."""

    def __init__(self, video, first, last, expected_sha256, *, timeout_seconds=120):
        if type(first) is not int or type(last) is not int or first < 0 or last < first:
            raise ValueError('Interval endpoints must be ordered nonnegative integer source-frame indices')
        if first > MAX_PREFIX_FRAMES:
            raise ValueError('Selected interval exceeds the 20000-frame sequential prefix bound')
        if last - first + 1 > MAX_SELECTED_FRAMES:
            raise ValueError('Selected interval exceeds 1800 frames')
        if not isinstance(expected_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', expected_sha256):
            raise ValueError('A lowercase SHA-256 source pin is required')
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= MAX_DEADLINE_SECONDS:
            raise ValueError('Decode deadline must be positive and no longer than 120 seconds')
        self.path = Path(video)
        self.first, self.last = first, last
        self.expected_sha256 = expected_sha256
        self.timeout_seconds = float(timeout_seconds)
        self._capture = None
        self._entered = False
        self._closed = False
        self._iteration_started = False
        self._selected_count = 0
        self._last_timestamp = None
        self._last_timestamp_index = None
        self._initial_zero_repeats = 0
        self._metadata = {}
        self._audit = {
            'scope': 'selected_interval', 'source_sha256': expected_sha256,
            'selected_interval': [first, last], 'selected_frames_expected': last - first + 1,
            'prefix_advanced_frames': 0, 'decoded_frames': 0,
            'prefix_grab_attempts': 0, 'selected_read_attempts': 0, 'selected_reads_succeeded': 0,
            'selected_frames_complete': False, 'source_complete': False,
            'selected_frames_complete_scope': 'Requested sequential decoder ordinal coverage only',
            'decoder_ordinals_complete': False,
            'encoded_frame_completeness': 'not_independently_verified', 'encoded_frame_skips': None,
            'prefix_pixels_retrieved': False, 'post_window_frames_decoded': 0,
            'source_frame_count_actual': None, 'post_window_frame_count_actual': None,
            'frame_id_basis': 'Sequential successful grabs from stream start, then one read per selected ordinal; no seek',
            'decoder_timestamps': [], 'decoder_positions_unavailable': 0,
            'decoder_timestamp_repeats': 0, 'state': 'not_opened',
            'deadline_scope': 'Checked before/after native calls; does not interrupt a blocked codec call',
            'limits': {'source_bytes': MAX_SOURCE_BYTES, 'source_metadata_frames': MAX_SOURCE_FRAMES,
                       'source_metadata_seconds': MAX_SOURCE_SECONDS, 'prefix_grabs': MAX_PREFIX_FRAMES,
                       'selected_frames': MAX_SELECTED_FRAMES, 'selected_seconds': MAX_SELECTED_SECONDS,
                       'deadline_seconds': self.timeout_seconds},
        }

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def audit(self):
        return deepcopy(self._audit)

    def _deadline(self):
        if time.monotonic() - self._started > self.timeout_seconds:
            raise TimeoutError('Selected-interval decode exceeded its deadline')

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _same_source(self):
        if self._signature(self.path) != self._source_signature:
            self._audit['source_stat_unchanged'] = False
            raise ValueError('Pinned source changed during selected-interval decoding')

    def _fail(self, error):
        self._audit.update(state='failed', failure={'type': type(error).__name__, 'message': str(error)})

    def _release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._closed = True
        if hasattr(self, '_started'):
            self._audit['elapsed_seconds'] = time.monotonic() - self._started

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError('IntervalVideo cannot be reopened')
        self._entered = True
        self._started = time.monotonic()
        try:
            self.path = self.path.resolve(strict=True)
            if not self.path.is_file() or self.path.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError('Source must be a local file no larger than 100 MB')
            self._source_signature = self._signature(self.path)
            hasher = sha256()
            with self.path.open('rb') as stream:
                while block := stream.read(1 << 20):
                    self._deadline()
                    hasher.update(block)
            self._deadline()
            if hasher.hexdigest() != self.expected_sha256:
                raise ValueError('Source checksum differs from the selected-interval pin')
            self._same_source()
            self._capture = cv2.VideoCapture(str(self.path))
            self._deadline()
            if not self._capture.isOpened():
                raise ValueError('Source video could not be opened')
            fps, raw_count, width, height = [float(self._capture.get(key)) for key in
                                            (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                                             cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
            if not all(math.isfinite(value) for value in (fps, raw_count, width, height)):
                raise ValueError('Source metadata must be finite')
            if not 1 <= fps <= 240 or raw_count != int(raw_count) or not 1 <= raw_count <= MAX_SOURCE_FRAMES:
                raise ValueError('Source FPS or advertised frame count exceeds bounds')
            if raw_count / fps > MAX_SOURCE_SECONDS:
                raise ValueError('Advertised source duration exceeds one hour')
            if width != int(width) or height != int(height) or not 64 <= width <= 4096 or not 64 <= height <= 2160:
                raise ValueError('Source dimensions exceed the 4096×2160 bound or 64-pixel minimum')
            if self.last >= int(raw_count):
                raise ValueError('Selected endpoint exceeds the advertised source frame count')
            selected = self.last - self.first + 1
            if selected / fps > MAX_SELECTED_SECONDS:
                raise ValueError('Selected interval exceeds 60 nominal seconds')
            self._metadata = {'fps': fps, 'frame_count': int(raw_count), 'width': int(width), 'height': int(height),
                              'frame_count_basis': 'advertised_metadata_not_full_decode',
                              'source_sha256': self.expected_sha256, 'source_bytes': self._source_signature[2],
                              'source_metadata_duration_s': raw_count / fps,
                              'selected_frame_count': selected, 'selected_nominal_duration_s': selected / fps,
                              'selected_nominal_start_time_s': self.first / fps,
                              'selected_nominal_end_time_s': self.last / fps}
            self._audit.update(state='opened', metadata_frames=int(raw_count),
                               post_window_frames_metadata=max(0, int(raw_count) - self.last - 1),
                               excluded_frames_metadata=int(raw_count) - selected,
                               outside_interval_completeness='Not inspected or certified; prefix advances do not retrieve prefix pixels',
                               fps=fps)
            period = 1000 / fps
            self._audit['timestamp_cfr_check'] = {
                'status': 'not_checked', 'nominal_frame_period_ms': period,
                'tolerance_ms': max(2., .1 * period),
                'tolerance_basis': 'Maximum of 2 ms quantization floor and 10% of nominal frame period',
                'scope': 'Reported timestamp cadence versus nominal FPS; not independent encoded-frame verification',
                'reported_timestamps': 0, 'unavailable_timestamps': 0,
                'zero_timestamps': 0, 'positive_timestamps': 0,
                'checked_intervals': 0, 'checked_intervals_spanning_missing': 0,
                'maximum_step_error_ms': 0., 'complete_timestamp_coverage': False,
            }
            self._same_source()
            return self
        except Exception as error:
            self._fail(error)
            self._release()
            raise

    def _position(self, expected):
        value = float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not math.isfinite(value) or value < 0 or (value == 0 and expected > 0):
            self._audit['decoder_positions_unavailable'] += 1
            return None
        if abs(value - expected) > .01:
            raise ValueError(f'Decoder position jumped or reordered: expected {expected}, reported {value}')
        return value

    def __iter__(self):
        if not self._entered or self._closed:
            raise RuntimeError('Use IntervalVideo inside its open context')
        if self._iteration_started:
            raise RuntimeError('Selected intervals support only one streaming pass')
        self._iteration_started = True
        return self._frames()

    def _timestamp_failure(self, message, details):
        self._audit['timestamp_cfr_check'].update(status='failed', failure=details)
        raise ValueError(message)

    def _check_timestamp(self, index, timestamp):
        check = self._audit['timestamp_cfr_check']
        if timestamp is None:
            check['unavailable_timestamps'] += 1
            check['status'] = 'unverified_missing_timestamps'
            return
        check['reported_timestamps'] += 1
        check['zero_timestamps'] += int(timestamp == 0)
        check['positive_timestamps'] += int(timestamp > 0)
        previous = self._last_timestamp
        if previous is not None:
            delta = timestamp - previous
            distance = index - self._last_timestamp_index
            expected = distance * check['nominal_frame_period_ms']
            details = {'source_frame': index, 'previous_source_frame': self._last_timestamp_index,
                       'decoder_time_ms': timestamp, 'previous_decoder_time_ms': previous,
                       'ordinal_distance': distance, 'delta_ms': delta, 'expected_delta_ms': expected,
                       'tolerance_ms': check['tolerance_ms']}
            if delta < -.001:
                self._timestamp_failure('Decoder presentation timestamps move backward within the selected interval', details)
            if timestamp == previous:
                self._audit['decoder_timestamp_repeats'] += 1
                if timestamp > 0:
                    self._timestamp_failure('Decoder repeats a usable presentation timestamp', details)
                self._initial_zero_repeats += 1
                check['status'] = 'pending_all_zero_backend_timestamps'
            else:
                if timestamp > 0 and self._initial_zero_repeats:
                    self._timestamp_failure('Repeated initial zero timestamps precede a usable decoder clock; cadence is unverified', details)
                error = abs(delta - expected)
                check['checked_intervals'] += 1
                check['checked_intervals_spanning_missing'] += int(distance > 1)
                check['maximum_step_error_ms'] = max(check['maximum_step_error_ms'], error)
                if error > check['tolerance_ms'] + 1e-6:
                    self._timestamp_failure('Decoder presentation timestamp gap disagrees with nominal constant-FPS cadence', details)
        self._last_timestamp, self._last_timestamp_index = timestamp, index

    def _finish_timestamp_audit(self):
        check = self._audit['timestamp_cfr_check']
        if not check['positive_timestamps']:
            status = 'unavailable_all_zero' if check['zero_timestamps'] else 'unavailable'
        elif check['unavailable_timestamps']:
            status = 'unverified_missing_timestamps'
        elif self._selected_count < 2:
            status = 'unverified_insufficient_samples'
        else:
            status = 'verified_cfr_cadence'
        check.update(status=status, complete_timestamp_coverage=(status == 'verified_cfr_cadence'))

    def _frames(self):
        try:
            self._audit['state'] = 'advancing_prefix'
            for index in range(self.first):
                self._deadline()
                self._audit['prefix_grab_attempts'] += 1
                ok = self._capture.grab()
                self._audit['prefix_advanced_frames'] += int(bool(ok))
                self._deadline()
                if not ok:
                    raise ValueError(f'EOF while advancing prefix at source frame {index}')
            self._audit['decoder_position_after_prefix'] = self._position(self.first)
            self._audit['state'] = 'decoding_selected'
            for index in range(self.first, self.last + 1):
                if self._closed:
                    raise RuntimeError('Selected interval was closed before streaming completed')
                self._deadline()
                self._audit['selected_read_attempts'] += 1
                ok, frame = self._capture.read()
                self._audit['selected_reads_succeeded'] += int(bool(ok))
                self._deadline()
                if not ok:
                    raise ValueError(f'EOF before selected endpoint at source frame {index}; expected through {self.last}')
                expected_shape = (self._metadata['height'], self._metadata['width'], 3)
                if not isinstance(frame, np.ndarray) or frame.shape != expected_shape or frame.dtype != np.uint8:
                    raise ValueError(f'Invalid or changing decoded BGR frame at source frame {index}')
                position = self._position(index + 1)
                timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = timestamp if math.isfinite(timestamp) and timestamp >= 0 else None
                record = {'source_frame': index, 'decoder_time_ms': timestamp,
                          'nominal_time_s': index / self._metadata['fps'],
                          'decoder_next_frame_position': position, 'accepted_for_stream': False}
                self._audit['decoder_timestamps'].append(record)
                self._check_timestamp(index, timestamp)
                record['accepted_for_stream'] = True
                self._selected_count += 1
                self._audit['decoded_frames'] = self._selected_count
                if index == self.last:
                    self._same_source()
                    self._finish_timestamp_audit()
                    self._audit.update(state='selected_complete', selected_frames_complete=True,
                                       decoder_ordinals_complete=True, selected_frame_ids_contiguous=True,
                                       selected_ordinal_skips=0,
                                       source_stat_unchanged=True)
                yield index, frame
        except Exception as error:
            self._fail(error)
            raise

    def __exit__(self, error_type, error, traceback):
        try:
            if error is not None:
                self._fail(error)
                return False
            if self._audit['state'] == 'failed':
                raise ValueError('Selected-interval decoding previously failed; no complete result')
            if self._selected_count != self.last - self.first + 1:
                raise ValueError('Selected interval was not completely consumed')
            self._deadline()
            self._same_source()
        except Exception as final_error:
            self._fail(final_error)
            raise
        finally:
            self._release()
        return False
