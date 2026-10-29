"""Prepare a source-pinned tracking prefix; never submit media or change pose evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.meta_segmentation import build_request
from weft.pipeline import verify_artifacts
from weft.video_io import webm_video_frame_count


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/exercise/jumping-jacks-burpees.webm'
POSES = ROOT/'artifacts/selective-crop-videos-03'
ROI = [260, 130, 100, 240]
COUNT = 350


def framehash(path):
    lines = path.read_text().splitlines()
    if '#tb 0: 1/1000' not in lines:
        raise ValueError('Unexpected framehash timebase; preserve before adapting the mapping')
    return [[field.strip() for field in line.split(',')] for line in lines if line and not line.startswith('#')]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-encoded', type=Path, help='Reuse a sealed encoded prefix with invalid final duration')
    parser.add_argument('--reuse-finalized', type=Path, help='Reuse completed media/hash preparation; finish mapping verification')
    args = parser.parse_args(); out = args.out.resolve()
    if args.reuse_encoded and args.reuse_finalized: raise ValueError('Choose one explicit reuse stage')
    if out.is_relative_to(POSES): raise ValueError('Use a fresh preparation directory')
    verified = verify_artifacts(POSES)
    evidence = json.loads((POSES/'jumping-jacks-burpees/selected/evidence.json').read_text())
    if digest(SOURCE) != evidence['source_sha256'] or webm_video_frame_count(SOURCE) != evidence['decoded_frames']:
        raise ValueError('Complete original video identity changed')
    reused = (args.reuse_encoded or args.reuse_finalized).resolve() if args.reuse_encoded or args.reuse_finalized else None
    reuse_verification = verify_artifacts(reused) if reused else None
    if reused:
        prior=json.loads((reused/'pre-run.json').read_text())
        if (out.is_relative_to(reused) or prior['source_sha256'] != evidence['source_sha256']
                or prior['requested_original_frames'] != [0, COUNT-1]):
            raise ValueError('Reuse requires the same original source and prefix')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'prepare_jacks_person_tracking.py')
    source_note = SOURCE.with_suffix('.source.json')
    if source_note.exists(): shutil.copyfile(source_note, out/'source-attribution.json')
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    started = time.monotonic()
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source': str(SOURCE),
        'source_sha256': evidence['source_sha256'], 'complete_source_frames': evidence['decoded_frames'],
        'pose_input_verification': verified, 'requested_original_frames': [0, COUNT-1], 'roi_frame_0_xywh': ROI,
        'roi_source': 'Original frame-0 visual subject selection, not comparison annotations',
        'hypothesis': 'Tracked person regions supply actual-frame crop proposals when short pose-neighbor context is absent',
        'not_claimed': ['whole-video Meta tracking', 'verified person identity', 'anatomy', 'contact', 'muscle activity'],
        'model': 'sam-3.1', 'pricing': {'url': 'https://dev.meta.ai/docs/pricing-rate-limits',
            'checked_at': local_time(), 'usd_per_1000_video_frames': .20, 'minimum_billed_frames': 50,
            'source_conflict': 'Generic introduction says no minimums; SAM-specific section states 50-frame minimum'},
        'estimated_one_request_usd': .07, 'conservative_reservation_usd': .15,
        'shared_total_budget_usd': 5, 'previous_committed_usd': .176152,
        'automatic_retries': 0, 'recurring_external_cost_usd': 0, 'files_api_uploads': 0,
        'encoding': 'H.264 CRF0, native YUV420P, no scaling/cropping/frame dropping within the explicit prefix; audio omitted',
        'ffmpeg_sha256': digest(ffmpeg), 'maximum_threads': 2, 'process_timeout_seconds': 60,
        'reused_encoded_input':str(reused) if reused else None, 'reused_input_verification':reuse_verification,
        'api_calls': 0})
    def execute(name, args):
        command = [str(ffmpeg), '-hide_banner', '-nostdin', '-n', *args]
        write_json(out/(name+'-command.json'), command)
        with (out/(name+'.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
    video = out/'jacks-prefix-lossless.mp4'
    unfinalized = out/'jacks-prefix-unfinalized.mp4'
    common = ['-map','0:v:0','-an','-sn','-dn']
    if args.reuse_finalized:
        shutil.copyfile(reused/'jacks-prefix-lossless.mp4',video)
        for name in ('source-frames.sha256','derivative-frames.sha256'):
            shutil.copyfile(reused/name,out/name)
    elif reused:
        shutil.copyfile(reused/'jacks-prefix-lossless.mp4',unfinalized)
        shutil.copyfile(reused/'source-frames.sha256',out/'source-frames.sha256')
    else:
        execute('encode', ['-threads','2','-copyts','-i',str(SOURCE),*common,
            '-vf',f'trim=end_frame={COUNT},setpts=PTS-STARTPTS','-frames:v',str(COUNT),
            '-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p','-threads:v','2',
            '-bf','0','-fps_mode','passthrough','-enc_time_base:v','demux','-video_track_timescale','1000',
            '-movflags','+faststart',str(unfinalized)])
    # A zero-duration trim endpoint can be hidden by MP4 edit lists; retain packets and original PTS.
    if not args.reuse_finalized:
        execute('remux', ['-ignore_editlist','1','-i',str(unfinalized),*common,'-c:v','copy',
            '-bsf:v','setts=duration=33','-video_track_timescale','1000','-movflags','+faststart',str(video)])
    for name, path, filters in (('source', SOURCE, ['-vf',f'trim=end_frame={COUNT}','-frames:v',str(COUNT)]),
                                ('derivative', video, [])):
        if args.reuse_finalized or (name == 'source' and reused): continue
        execute(name+'-hash', ['-threads','1','-copyts','-i',str(path),*common,*filters,
            '-c:v','rawvideo','-pix_fmt','yuv420p','-threads:v','1','-fps_mode','passthrough',
            '-enc_time_base:v','demux','-f','framehash','-hash','sha256',str(out/(name+'-frames.sha256'))])
    before, after = framehash(out/'source-frames.sha256'), framehash(out/'derivative-frames.sha256')
    if len(before) != COUNT or len(after) != COUNT:
        raise ValueError('Explicit prefix frame count differs; no request is prepared')
    offset = int(before[0][2])-int(after[0][2])
    if any(int(a[2])-offset != int(b[2]) or a[5] != b[5] for a,b in zip(before,after,strict=True)):
        raise ValueError('Prefix timestamp or native YUV pixels changed')
    cap, derivative = cv2.VideoCapture(str(SOURCE)), cv2.VideoCapture(str(video))
    mapping = []; decoder_offset = None
    try:
        for frame in range(COUNT):
            ok1,a=cap.read(); ok2,b=derivative.read()
            if not ok1 or not ok2 or a.shape != b.shape: raise ValueError('Source/derivative decoder coverage differs')
            stamps = [float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000 for capture in (cap,derivative)]
            if decoder_offset is None: decoder_offset = stamps[0]-stamps[1]
            if abs(stamps[0]-stamps[1]-decoder_offset) > 1e-7: raise ValueError('Decoder timestamp mapping changed')
            mapping.append({'source_frame':frame, 'relative_frame':frame, 'source_time_s':frame/evidence['fps'],
                'source_decoder_time_s':stamps[0], 'derivative_decoder_time_s':stamps[1],
                'source_pts_ms':int(before[frame][2]), 'derivative_pts_ms':int(after[frame][2]),
                'source_yuv_sha256':before[frame][5], 'derivative_yuv_sha256':after[frame][5],
                'source_bgr_sha256':sha256(a.tobytes()).hexdigest(), 'derivative_bgr_sha256':sha256(b.tobytes()).hexdigest(),
                'identical_bgr_pixels':bool(np.array_equal(a,b))})
            if frame == 0:
                if not cv2.imwrite(str(out/'initialization-source.png'),a): raise OSError('Cannot save initialization source')
                x,y,w,h=ROI; cv2.rectangle(a,(x,y),(x+w-1,y+h-1),(0,215,255),2)
                if not cv2.imwrite(str(out/'initialization-box.png'),a): raise OSError('Cannot save initialization selection')
        if derivative.read()[0]: raise ValueError('Derivative contains a hidden suffix')
    finally:
        cap.release(); derivative.release()
    write_json(out/'frame-mapping.json', {'original_source_sha256':evidence['source_sha256'],
        'derivative_sha256':digest(video), 'original_full_frame_count':evidence['decoded_frames'],
        'initial_pts_offset_ms':offset, 'initial_decoder_offset_s':decoder_offset,
        'selected_original_interval':[0,COUNT-1], 'frames':mapping})
    request = build_request(video, expected_sha256=digest(video), frame_count=COUNT,
                            width=evidence['width'], height=evidence['height'], roi=ROI)
    write_json(out/'report.json', {'status':'ready_local_preparation_only', 'created_at':local_time(),
        'source_sha256':evidence['source_sha256'], 'derivative_sha256':digest(video), 'derivative_bytes':video.stat().st_size,
        'request_bytes':len(request), 'frame_count':COUNT, 'dimensions':[evidence['width'],evidence['height']],
        'roi_frame_0_xywh':ROI, 'exact_native_yuv_and_relative_timestamps':True,
        'initial_pts_offset_ms':offset, 'initial_decoder_offset_s':decoder_offset,
        'identical_bgr_frames':sum(row['identical_bgr_pixels'] for row in mapping),
        'estimated_request_usd':.07, 'reservation_usd':.15, 'api_calls':0,
        'runtime_seconds':time.monotonic()-started,
        'scope':'Explicit 350-frame prefix for a source-region recovery test; all 1,356 original frames remain in the pose input'})
    if verify_artifacts(POSES) != verified: raise ValueError('Original pose evidence changed')
    if reused and verify_artifacts(reused) != reuse_verification: raise ValueError('Saved encoding input changed')
    print(json.dumps({'report':json.loads((out/'report.json').read_text()),'verification':seal_suite(out)},indent=2))


if __name__ == '__main__': main()
