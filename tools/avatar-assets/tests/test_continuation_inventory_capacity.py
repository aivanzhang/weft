"""Finite metadata/hash limits; large logical inputs are sparse or scaled analytic fixtures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import pipeline_continuation as cont


class InventoryCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_measured_three_attempt_capacity_remains_finite(self):
        self.assertEqual((cont.MAX_FILES,cont.MAX_BYTES,cont.MAX_ENTRIES),(4096,1024**3,8192))
        self.assertEqual((cont.MAX_FILE_BYTES,cont.INVENTORY_SECONDS),(64*1024**2,20))
        self.assertLess(1175,cont.MAX_FILES);self.assertLess(641261780,cont.MAX_BYTES)

    def test_exact_file_count_is_admitted_and_next_file_rejected(self):
        for index in range(cont.MAX_FILES):(self.root/str(index)).touch()
        rows=cont._inventory(self.root);self.assertEqual(len(rows),cont.MAX_FILES)
        self.assertEqual(sum(row['bytes'] for row in rows),0)
        (self.root/'extra').touch()
        with self.assertRaisesRegex(ValueError,'4096 files'):cont._inventory(self.root)

    def test_exact_entry_count_is_admitted_and_next_directory_rejected(self):
        for index in range(cont.MAX_ENTRIES):(self.root/str(index)).mkdir()
        self.assertEqual(cont._inventory(self.root),[])
        (self.root/'extra').mkdir()
        with self.assertRaisesRegex(ValueError,'inventory exceeded bounds'):cont._inventory(self.root)

    def test_aggregate_bytes_count_all_files_and_do_not_depend_on_file_limit(self):
        (self.root/'a').write_bytes(b'abc');(self.root/'b').write_bytes(b'defg')
        with patch.object(cont,'MAX_BYTES',7):
            self.assertEqual(sum(row['bytes'] for row in cont._inventory(self.root)),7)
            (self.root/'c').write_bytes(b'h')
            with self.assertRaisesRegex(ValueError,'1GiB'):cont._inventory(self.root)

    def test_exact_64mib_file_streams_and_larger_file_is_rejected_before_open(self):
        path=self.root/'input.glb'
        with path.open('wb') as stream:stream.truncate(cont.MAX_FILE_BYTES)
        row=cont._inventory(self.root)[0];self.assertEqual(row['bytes'],cont.MAX_FILE_BYTES)
        with path.open('wb') as stream:stream.truncate(cont.MAX_FILE_BYTES+1)
        with patch.object(Path,'open',side_effect=AssertionError('Oversized payload must not be read')):
            with self.assertRaisesRegex(ValueError,'64MiB'):cont._inventory(self.root)

    def test_deadline_still_stops_before_open_and_during_chunked_hashing(self):
        (self.root/'data').write_bytes(b'x'*(2*1024**2))
        with patch.object(cont.time,'monotonic',side_effect=[0,20]):
            with self.assertRaisesRegex(ValueError,'inventory exceeded bounds'):cont._inventory(self.root)
        with patch.object(cont.time,'monotonic',side_effect=[0,0,0,20]):
            with self.assertRaisesRegex(ValueError,'exceeded deadline'):cont._inventory(self.root)

    def test_symbolic_links_are_not_admitted(self):
        (self.root/'target').write_bytes(b'Bounded original')
        (self.root/'linked').symlink_to(self.root/'target')
        with self.assertRaisesRegex(ValueError,'contains links'):cont._inventory(self.root)


if __name__=='__main__':unittest.main()
