import struct
import unittest

from PIL import Image

from prepare_hwdb import fit_image, iter_gnt, split_writers, writer_id


class HWDBTests(unittest.TestCase):
    def test_gnt_label_size_and_record_boundary(self):
        records = b''
        for char in ('日', '田'):
            records += struct.pack('<I2sHH', 16, char.encode('gb2312'), 3, 2) + bytes(range(6))
        parsed = list(iter_gnt(records))
        self.assertEqual([r[1] for r in parsed], [0, 16])
        self.assertEqual([r[2].decode('gb2312') for r in parsed], ['日', '田'])
        self.assertEqual(parsed[1][3:], (3, 2, bytes(range(6))))

    def test_rejects_truncation_or_inconsistent_dimensions(self):
        record = struct.pack('<I2sHH', 16, b'xx', 3, 2) + bytes(range(6))
        for invalid in (record[:3], record[:-1], struct.pack('<I2sHH', 17, b'xx', 3, 2) + bytes(range(7))):
            with self.assertRaises(ValueError):
                list(iter_gnt(invalid))

    def test_writer_groups_are_reproducible_and_isolated(self):
        train = [str(i) for i in range(240)]
        test = [str(i) for i in range(240, 300)]
        groups = split_writers(train, test, 2026, 0.1)
        self.assertEqual({k: len(v) for k, v in groups.items()}, {'train': 216, 'val': 24, 'test': 60})
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse((groups['train'] | groups['val']) & groups['test'])
        self.assertEqual(groups, split_writers(train[::-1], test, 2026, 0.1))
        with self.assertRaises(ValueError):
            split_writers(train, ['0'], 2026, 0.1)

    def test_image_preserves_background_and_geometry(self):
        result = fit_image(Image.new('L', (20, 10), color=0))
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(result.getpixel((0, 0)), 255)
        self.assertEqual(result.getpixel((32, 32)), 0)
        self.assertEqual(result.getpixel((32, 2)), 255)

    def test_writer_filename_parsing(self):
        self.assertEqual(writer_id('Gnt1.1TrainPart1/1001-c.gnt'), '1001')
        with self.assertRaises(ValueError):
            writer_id('unexpected.gnt')
