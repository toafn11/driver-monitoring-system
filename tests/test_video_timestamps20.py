import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from training.video_timestamps20 import decoded_timestamps
from training.prepare20 import extract_video
from test_prepare20 import video, DetectorFixture


class TimestampTest(unittest.TestCase):
    def probe(self, times, stderr=""):
        output = json.dumps(dict(frames=[dict(best_effort_timestamp_time=t) for t in times]))
        with patch('shutil.which', return_value='ffprobe'), patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=output, stderr=stderr)):
            return decoded_timestamps('fixture.mov')

    def test_pts_validation(self):
        np.testing.assert_allclose(self.probe([10,10.031,10.065]), [0,.031,.065])
        for times in ([0,0,.03], [0,.04,.03], [0,float('nan')]):
            with self.assertRaises(ValueError): self.probe(times)
        with self.assertRaises(ValueError): self.probe([0,.03], 'decode error')

    def test_sampling_uses_pts_not_container_fps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'source.avi'; video(path)  # container says 30 FPS
            model=root/'model.task'; model.write_bytes(b'fixture')
            times=np.arange(120)/25  # independent decoded clock
            with patch('core.face_analyzer.FaceAnalyzer', DetectorFixture), patch('training.video_timestamps20.decoded_timestamps', return_value=times):
                raw, sampled, valid=extract_video(path,model,10,root/'cache')
            self.assertEqual(len(raw),48)
            self.assertGreater(sampled[-1],4.6)
            self.assertTrue(np.isin(sampled,times).all())

    def test_disagreeing_decoders_never_write_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'source.avi'; video(path)
            model=root/'model.task'; model.write_bytes(b'fixture')
            with patch('core.face_analyzer.FaceAnalyzer', DetectorFixture), patch('training.video_timestamps20.decoded_timestamps', return_value=np.arange(121)/30):
                with self.assertRaisesRegex(ValueError,'Decoder count mismatch'):
                    extract_video(path,model,10,root/'cache')
            self.assertEqual(list((root/'cache').glob('*.npz')),[])


if __name__ == '__main__': unittest.main()
