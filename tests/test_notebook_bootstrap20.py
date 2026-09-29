"""Check the shipped notebook really replaces cached project imports."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class BootstrapTest(unittest.TestCase):
    def test_stale_modules_replaced_and_changed_source_rejected(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'kaggle_01_extract20.ipynb').read_text(encoding='utf-8'))
        source = ''.join(notebook['cells'][1]['source'])
        activation = source.split('# BEGIN VERIFIED PROJECT ACTIVATION\n')[1].split('# END VERIFIED PROJECT ACTIVATION')[0]
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(io.BytesIO(subprocess.check_output(
                    ['git','archive','--format=zip','a637d62'], cwd=root))) as z:
                z.extractall(d)
            script = '''import sys, types
from pathlib import Path
CODE = Path(sys.argv[1])
fake = types.ModuleType('training.preflight20')
fake.__file__ = '/old/preflight20.py'
sys.modules['training.preflight20'] = fake
activation = sys.argv[2]
exec(activation)
assert sys.modules['training.preflight20'] is not fake
p = CODE/'training/preflight20.py'
p.write_bytes(p.read_bytes() + b'\\n# changed\\n')
try:
    exec(activation)
except RuntimeError as e:
    assert 'integrity mismatch' in str(e)
else:
    raise AssertionError('Modified source was accepted')
'''
            result = subprocess.run([sys.executable,'-c',script,d,activation], capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
