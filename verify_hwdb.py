"""验证实际子集的来源清单、图像、缓存、类别计数与书写人隔离。"""
from collections import Counter
import json

import numpy as np
from PIL import Image

from data import DATA_ROOT, hwdb_metadata, prepare_split, prepare_test
from download_hwdb import sha256_file
from prepare_hwdb import fit_image


def main():
    root = DATA_ROOT / 'hwdb10'
    info = hwdb_metadata()
    rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    sources = [(r['archive'], r['member'], r['record_index']) for r in rows]
    assert len(sources) == len(set(sources)), 'Duplicate original sample identity'
    groups = {k: set(v) for k, v in info['writer_ids'].items()}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    pixel_splits = {}
    for split in ('train', 'val', 'test'):
        assert sha256_file(root / f'{split}.npz') == info['array_sha256'][split]
        subset = [r for r in rows if r['split'] == split]
        assert len(subset) == info['sample_counts'][split]
        assert {str(k): v for k, v in Counter(r['label'] for r in subset).items()} == info['class_counts'][split]
        with np.load(root / f'{split}.npz', allow_pickle=False) as data:
            assert data['images'].shape == (len(subset), 64, 64)
            assert data['images'].dtype == np.uint8
            assert set(data['labels'].tolist()) == set(range(10))
            assert {r['array_index'] for r in subset} == set(range(len(subset)))
            for row in subset:
                assert row['writer_id'] in groups[split]
                assert row['official_partition'] == ('test' if split == 'test' else 'train')
                assert row['character'] == info['classes'][row['label']]
                assert bytes.fromhex(row['gb2312_hex']).decode('gb2312') == row['character']
                path = root / row['image']
                assert sha256_file(path) == row['png_sha256']
                with Image.open(path) as image:
                    assert image.mode == 'L' and image.size == (row['width'], row['height'])
                    np.testing.assert_array_equal(np.asarray(fit_image(image)), data['images'][row['array_index']])
                assert int(data['labels'][row['array_index']]) == row['label']
                previous = pixel_splits.setdefault(row['pixel_sha256'], split)
                assert previous == split, 'Exact image duplicate across splits'
    train, val, _ = prepare_split('hwdb10')
    test, classes = prepare_test('hwdb10')
    result = {'passed': True, 'samples_verified': len(rows),
              'train': len(train), 'val': len(val), 'test': len(test),
              'classes': classes, 'writer_counts': info['writer_counts'],
              'checks': ['source identities unique', 'writers disjoint', 'official test boundary preserved',
                         'all 10 classes in every split', 'all PNG hashes', 'all cached images reconstructed',
                         'all label encodings', 'NPZ hashes and loader integration', 'no cross-split exact duplicates']}
    (root / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
