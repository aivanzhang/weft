"""Full-size visual evidence from native poses, exported skin motion and encoded media."""

from pathlib import Path
import shutil
import sys

import av
from PIL import Image, ImageDraw

from avatar_assets.safety import artifacts, contained, local_file, new_directory, read_json, require, sha256, slug, write_json

TILE = 720
LABEL = 28
VIEWS = ('front', 'side', 'back', 'three_quarter')


def select_indices(count):
    require(type(count) is int and 2 <= count <= 480, 'Unbounded or static evidence sequence')
    return sorted({round(i*(count-1)/3) for i in range(4)})


def decoded_poses(path, expected):
    """Decode once with strict frame bounds; never substitute source PNGs for delivered video."""
    path = local_file(path, 64*1024**2)
    count = expected['frames']
    require(type(count) is int and 2 <= count <= 480 and expected['width']*expected['height'] <= 1024**2,
            'Video evidence exceeds decode bounds')
    wanted, frames = set(select_indices(count)), {}
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1 and not container.streams.audio, 'Expected one silent video stream')
        stream = container.streams.video[0]
        require((stream.width, stream.height) == (expected['width'], expected['height']), 'Video dimensions changed')
        decoded = 0
        for index, frame in enumerate(container.decode(stream)):
            require(index < count and frame.width*frame.height <= 1024**2, 'Video frame bound changed')
            if index in wanted:
                frames[index] = frame.to_image().convert('RGB')
            decoded += 1
    require(decoded == count and set(frames) == wanted, 'Video frame count changed')
    return [(index, frames[index]) for index in sorted(frames)]


def _read_image(path):
    path = local_file(path, 8*1024**2)
    with Image.open(path) as image:
        require(image.width*image.height <= 16_777_216 and getattr(image, 'n_frames', 1) == 1,
                'Visual evidence exceeds image bounds')
        return image.convert('RGB')


def _tile(sheet, image, x, y, label):
    # Smaller media stays visibly smaller; padding is not additional resolution.
    image = image.copy()
    image.thumbnail((TILE, TILE))
    sheet.paste(image, (x+(TILE-image.width)//2, y+LABEL+(TILE-image.height)//2))
    ImageDraw.Draw(sheet).text((x+8, y+6), label, fill='black')


def build_evidence(pack, exported, media, output, reference=None):
    pack, exported, media = map(lambda p: Path(p).resolve(), (pack, exported, media))
    out = new_directory(output)
    finish = read_json(pack/'finish-report.json')
    encoded = read_json(media/'manifest.json')
    require(encoded.get('parent_3d_manifest_sha256') == sha256(pack/'manifest.json'),
            'Delivered media belongs to a different model')
    states = finish['states']
    require(1 <= len(states) <= 8 and {s['id'] for s in states} == {s['id'] for s in encoded['clips']},
            'Native and delivered states differ')
    images, coverage, sources = [], [], {}

    def source(path):
        path = local_file(path, 64*1024**2)
        sources[str(path)] = sha256(path)
        return path

    def record(path, kind, state_ids=()):
        with Image.open(path) as im:
            require(im.width*im.height <= 16_777_216, 'Evidence montage exceeds image bounds')
            width, height = im.size
        require(path.stat().st_size <= 8*1024**2, 'Evidence image exceeds agent attachment limit')
        identifier = 'evidence-'+path.stem
        images.append({'id': identifier, 'path': str(path.relative_to(out)), 'kind': kind,
                       'state_ids': list(state_ids), 'width': width, 'height': height, 'sha256': sha256(path)})
        return identifier, width, height

    for representation, root in (('native', pack/'views'), ('exported', exported)):
        sheet = Image.new('RGB', (TILE*2, (TILE+LABEL)*2), 'white')
        for index, view in enumerate(VIEWS):
            name = 'three-quarter' if representation == 'exported' and view == 'three_quarter' else view
            path = source(root/(name+'.png'))
            image = _read_image(path)
            require(min(image.size) >= TILE, 'Multiview review requires actual720px renders')
            _tile(sheet, image, index%2*TILE, index//2*(TILE+LABEL), representation+' / '+view)
        path = out/(representation+'-views.png')
        sheet.save(path)
        identifier, _, _ = record(path, 'multiview')
        coverage.extend({'id': identifier+'-'+view, 'kind': 'view', 'representation': representation,
                         'view': view, 'width': TILE, 'height': TILE} for view in VIEWS)

    batch_size = 1 if len(states) <= 4 else 2
    pose_rows = 2 if batch_size == 1 else 1
    rows_per_state = 2*pose_rows+1
    for first in range(0, len(states), batch_size):
        batch = states[first:first+batch_size]
        sheet = Image.new('RGB', (TILE*4, len(batch)*rows_per_state*(TILE+LABEL)), 'white')
        for row, state in enumerate(batch):
            identifier = slug(state['id'])
            for lane, (representation, root) in enumerate((('native', pack), ('exported', exported))):
                poses = sorted((root/'renders'/identifier).glob('*.png'))
                require(2 <= len(poses) <= 8, 'Missing bounded native/exported motion poses')
                indices = list(range(len(poses))) if batch_size == 1 else select_indices(len(poses))
                for tile, index in enumerate(indices):
                    image = _read_image(source(poses[index]))
                    require(min(image.size) >= TILE, 'State review requires actual720px poses')
                    second = state['seconds'][index]
                    _tile(sheet, image, tile%4*TILE, (row*rows_per_state+lane*pose_rows+tile//4)*(TILE+LABEL),
                          f'{identifier} / {representation} / {second:.3f}s')
            clip = next(c for c in encoded['clips'] if c['id'] == identifier)
            video = source(contained(media, clip['path']))
            require(sha256(video) == clip['inspection']['sha256'], 'Delivered video changed')
            for column, (index, image) in enumerate(decoded_poses(video, clip['inspection'])):
                _tile(sheet, image, column*TILE, (row*rows_per_state+2*pose_rows)*(TILE+LABEL),
                      f'{identifier} / delivered video / {index/24:.3f}s / {image.width}px')
        path = out/f'motion-{first//batch_size:02d}.png'
        sheet.save(path)
        identifier, width, height = record(path, 'native_exported_and_delivered_motion', [s['id'] for s in batch])
        for representation in ('native', 'exported'):
            coverage.append({'id': identifier+'-'+representation, 'kind': 'state', 'representation': representation,
                             'state_ids': [s['id'] for s in batch], 'width': width, 'height': height,
                             'pose_width': TILE, 'pose_height': TILE})

    portrait = source(media/'portrait.webp')
    target = out/'delivered-portrait.webp'
    shutil.copyfile(portrait, target)
    identifier, width, height = record(target, 'delivered_portrait')
    coverage.append({'id': identifier, 'kind': 'portrait', 'representation': 'native', 'width': width, 'height': height})
    if reference:
        path = source(reference)
        target = out/('selected-reference'+path.suffix.lower())
        shutil.copyfile(path, target)
        record(target, 'user_reference')
    require(len(images) <= 8 and sum((out/i['path']).stat().st_size for i in images) <= 32*1024**2,
            'Final evidence exceeds the bounded agent attachment allowance')
    report = {'schema_version': 1, 'kind': 'full_resolution_review_evidence', 'status': 'prepared_for_review',
              'sources': sources, 'model_manifest_sha256': sha256(pack/'manifest.json'),
              'exported_manifest_sha256': sha256(exported/'manifest.json'),
              'media_manifest_sha256': sha256(media/'manifest.json'), 'images': images, 'coverage': coverage,
              'scope': 'Up to eight actual poses per representation for one to four states; four selected poses for five to eight states. Four decoded delivered video samples per state. Not all-frame visual, physics or loop certification.',
              'quality_equivalence_verified': False, 'artifacts': artifacts(out)}
    write_json(out/'manifest.json', report)
    return report


if __name__ == '__main__':
    args = sys.argv[1:]
    require(4 <= len(args) <= 5, 'Use model-pack exported-pack media-pack new-output [reference]')
    build_evidence(*args)
