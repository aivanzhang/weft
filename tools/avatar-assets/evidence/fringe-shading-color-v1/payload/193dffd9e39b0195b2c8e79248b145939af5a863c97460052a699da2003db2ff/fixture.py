"""Unchanged analytic fixture extracted from the pinned allocation screen."""
import numpy as np

def fixture(counts, reverse_diagonal=False):
    # Extruding the same polygonal arc keeps all three surfaces' actual areas equal.
    columns, radius, band_height = 40, .12, .216
    theta = np.linspace(-np.pi/2, np.pi/2, columns+1)
    z = np.concatenate([np.linspace(-.324+i*band_height, -.324+(i+1)*band_height,
                                   count+1)[int(i > 0):] for i, count in enumerate(counts)])
    points = np.array([(radius*np.sin(t), -radius*np.cos(t), h) for h in z for t in theta])
    normal = np.array([(np.sin(t), -np.cos(t), 0.) for _ in z for t in theta])
    uv = np.array([(x/columns, 1-(h+.324)/.648) for h in z for x in range(columns+1)])
    faces = []
    for row in range(len(z)-1):
        for column in range(columns):
            a = row*(columns+1)+column
            b, c, d = a+1, a+columns+2, a+columns+1
            if reverse_diagonal and (row+column) % 2 == 0:
                faces.extend(((a, b, d), (b, c, d)))
            else:
                faces.extend(((a, b, c), (a, c, d)))
    faces = np.asarray(faces, dtype=np.int32)
    weights = np.zeros((len(points), 4))
    weights[:, 1] = np.clip((points[:, 2]+.04)/.08, 0, 1)
    weights[:, 0] = 1-weights[:, 1]
    pixels = np.array([[[255, 0, 0], [0, 255, 0]],
                       [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    nf = len(faces)
    return {'points': points, 'faces': faces, 'corner_normals': normal[faces], 'uv0': uv[faces],
            'materials': np.zeros(nf, dtype=np.int32), 'parts': np.zeros(nf, dtype=np.int32),
            'fur_faces': np.ones(nf, dtype=bool), 'non_fur_faces': np.zeros(nf, dtype=bool),
            'closure_faces': np.zeros(nf, dtype=bool),
            'joints': np.tile([0, 1, 0, 0], (len(points), 1)).astype(np.uint16), 'weights': weights,
            'flow_directions': np.tile([0., 0., -1.], (len(points), 1)),
            'textures': {0: {'pixels': pixels, 'factor': np.ones(3)},
                         1: {'pixels': pixels, 'factor': np.ones(3)}},
            'design_views': np.array([[np.sin(t), -np.cos(t), .1]
                                     for t in np.linspace(-np.pi, np.pi, 8, endpoint=False)])}
