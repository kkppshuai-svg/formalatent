import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from structure_vae import train, StructureVAE
from test_structure_vae import samples

class JavaScriptParityTests(unittest.TestCase):
    def test_js_encoder_matches_python(self):
        model=train(samples(),epochs=5)
        query={'tokens':['common','family:1']}
        expected=StructureVAE(model).encode([query])[0]
        module=Path(__file__).resolve().parents[1]/'integrations/ai-cad/structure-vae-runtime.mjs'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.json';p.write_text(json.dumps({'model':model,'tokens':query['tokens']}))
            code="import fs from 'node:fs'; import {encodeStructureTokens} from '"+module.as_uri()+"'; const {model,tokens}=JSON.parse(fs.readFileSync(process.argv[1],'utf8')); console.log(JSON.stringify(encodeStructureTokens(model,tokens)));"
            result=subprocess.check_output(['node','--input-type=module','-e',code,str(p)],text=True)
            np.testing.assert_allclose(json.loads(result),expected,atol=1e-12)

if __name__=='__main__': unittest.main()
