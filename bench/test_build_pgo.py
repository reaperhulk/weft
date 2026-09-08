"""Input-integrity checks for the release PGO builder; no compiler required."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from build_pgo import prepare_clip


class CorpusIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.raw = self.data / "clip.rgba"
        self.raw.write_bytes(bytes([255, 0, 0, 255]))
        self.clip = dict(name="clip", width=1, height=1, frames=1,
                         rgba_sha256=hashlib.sha256(self.raw.read_bytes()).hexdigest())

    def test_pinned_raw_input(self):
        self.assertEqual(prepare_clip(self.data, self.data, self.clip), self.raw)

    def test_corrupted_raw_input(self):
        self.raw.write_bytes(bytes([0, 255, 0, 255]))
        with self.assertRaisesRegex(ValueError, "RGBA checksum mismatch"):
            prepare_clip(self.data, self.data, self.clip)

    def test_wrong_frame_count(self):
        self.clip["frames"] = 2
        with self.assertRaisesRegex(ValueError, "Unexpected RGBA length"):
            prepare_clip(self.data, self.data, self.clip)

    def test_corrupted_mp4_rejected_before_decoding(self):
        self.raw.unlink()
        (self.data / "clip.mp4").write_bytes(b"corrupted")
        self.clip["mp4_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "MP4 checksum mismatch"):
            prepare_clip(self.data, self.data, self.clip)

    def test_path_outside_corpus(self):
        self.clip["name"] = "../clip"
        with self.assertRaisesRegex(ValueError, "Invalid clip name"):
            prepare_clip(self.data, self.data, self.clip)


if __name__ == "__main__":
    unittest.main()
