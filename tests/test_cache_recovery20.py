import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_prepare20 import video, DetectorFixture
from training.prepare20 import extract_video
from training.recover_cache20 import sampled_indices, verify_geometry, recover_shard0


class CacheRecoveryTest(unittest.TestCase):
    def test_full_legacy_migration_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); old=root/'legacy'; old.mkdir(); new=root/'new'
            model=root/'model.task'; model.write_bytes(b'model')
            paths=[(root/f'{i:03}'/'source.mp4').as_posix() for i in range(121)]
            paths[0]=(root/'000/Fold1_part2/Fold1_part2/11/0.mp4').as_posix()
            records=[]; clocks={}
            for number,path in enumerate(paths[4::4]):
                p=Path(path);p.parent.mkdir(parents=True);p.write_bytes(str(number).encode())
                pts=np.arange(120)/(30+number/100)
                clocks[path]=pts; times=pts[sampled_indices(pts,10)]
                np.savez_compressed(old/f'legacy{number}.npz',raw=np.zeros((len(times),9)),times=times,valid=np.ones(len(times),bool))
                records.append(dict(path=path,status='ok',frames=len(times),valid_fraction=1.0))
            (old/'progress_0.json').write_text(json.dumps(records))
            rows=[dict(path=p) for p in paths]
            with patch('training.recover_cache20.read_manifest',return_value=rows), \
                 patch('training.recover_cache20.decoded_timestamps',side_effect=lambda p:clocks[p]), \
                 patch('training.recover_cache20.verify_geometry',return_value=32) as geometry:
                recover_shard0('manifest',model,new,[old])
                self.assertEqual(geometry.call_count,30)
            self.assertEqual(len(list(new.glob('*.npz'))),30)
            with patch('training.recover_cache20.read_manifest',return_value=rows), \
                 patch('training.recover_cache20.decoded_timestamps',side_effect=AssertionError('Do not repeat migration')):
                recover_shard0('manifest',model,new,[old])

    def test_remounted_mtime_requires_matching_content(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'source.avi'; video(path)
            model=root/'model.task'; model.write_bytes(b'fixture')
            with patch('core.face_analyzer.FaceAnalyzer',DetectorFixture), patch('training.video_timestamps20.decoded_timestamps',return_value=np.arange(120)/30):
                raw,times,valid=extract_video(path,model,10,root/'old')
            stat=path.stat(); os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+5000000000))
            with patch('training.video_timestamps20.decoded_timestamps',side_effect=AssertionError('Must reuse cached data')):
                actual=extract_video(path,model,10,root/'new',[root/'old'],cache_only=True)
            np.testing.assert_array_equal(raw,actual[0])
            with path.open('r+b') as f: f.write(b'BAD!')
            with self.assertRaises(FileNotFoundError):
                extract_video(path,model,10,root/'new',[root/'old'],cache_only=True)

    def test_geometry_verification_rejects_wrong_video_features(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'source.avi'; video(path)
            model=root/'model.task'; model.write_bytes(b'fixture')
            with patch('core.face_analyzer.FaceAnalyzer',DetectorFixture), patch('training.video_timestamps20.decoded_timestamps',return_value=np.arange(120)/30):
                raw,times,valid=extract_video(path,model,10,root/'old')
                indices=sampled_indices(np.arange(120)/30,10)
                self.assertEqual(verify_geometry(path,model,raw,valid,indices),32)
                wrong=raw.copy();wrong[:,0]+=0.1
                with self.assertRaisesRegex(ValueError,'does not match'):
                    verify_geometry(path,model,wrong,valid,indices)

    def test_sampler_skips_duplicate_without_time_compression(self):
        times=np.array([0,.05,.1,.1,.15,.2,.5,.55,.6])
        self.assertEqual(sampled_indices(times,10).tolist(),[0,2,5,6,8])


if __name__=='__main__': unittest.main()
