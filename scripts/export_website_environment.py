"""Export one verified run; adding scenes never requires changing the viewer."""
import argparse
import json
from pathlib import Path

from weft.gallery import result_record
from weft.pipeline import verify_artifacts
from weft.source import SAMPLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    verify_artifacts(args.run)
    record = result_record(args.run)
    if record['source_sha256'] != SAMPLE['sha256']:
        raise ValueError('Only the approved CC0 sample may enter the public gallery through this exporter')
    record.update(video='../data/rolling-ball.mp4',poster='environments/rolling-ball.jpg',
                  download='environments/rolling-ball.json',attribution=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'catalog.js').write_text('globalThis.WeftEnvironments = '+json.dumps([record],allow_nan=False)+';\n')
    (args.out/'rolling-ball.json').write_text(json.dumps({'world':record['models'][0]['world'],
        'task':json.loads((args.run/'task.json').read_text()),'source_sha256':record['source_sha256'],
        'manifest_sha256':record['manifest_sha256'],'note':'Use world with Weft RollingEnv. Push controls and the parking goal are authored extensions.'},indent=2)+'\n')
    import cv2
    cap = cv2.VideoCapture(str(args.run/'source.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,record['models'][0]['world']['source_frame'])
    ok,frame = cap.read(); cap.release()
    if not ok: raise ValueError('Source poster frame could not be read')
    cv2.imwrite(str(args.out/'rolling-ball.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,88])
    print(json.dumps({'scenes':1,'models':len(record['models']),'manifest':record['manifest_sha256']}))


if __name__=='__main__': main()
