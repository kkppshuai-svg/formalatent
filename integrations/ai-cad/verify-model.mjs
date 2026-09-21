import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [root, artifacts] = process.argv.slice(2).map(p => path.resolve(p));
if (!root || !artifacts) throw new Error('Usage: node verify-model.mjs AICAD_ROOT ARTIFACT_DIR');
const { CadLatentTrainingManager } = await import(pathToFileURL(path.join(root, 'cad-latent-training.js')));
const { rankCadLatentModelSamples } = await import(pathToFileURL(path.join(root, 'cad-latent-memory.js')));
const runtimeDir = await fs.mkdtemp(path.join(os.tmpdir(), 'formalatent-integration-'));
const manager = new CadLatentTrainingManager({rootDir:root, runtimeDir,
  seedStructureDataset:path.join(artifacts,'train.jsonl'), seedBrepDataset:path.join(artifacts,'train.jsonl'),
  seedStructureModel:path.join(artifacts,'structure-vae.json'),seedBrepModel:path.join(artifacts,'geometry-vae.json')});
await manager.init();
const model = await manager.plannerModel();
assert.equal(model.format, 'formalatent-structure-vae-v1');
const expected = JSON.parse(await fs.readFile(path.join(artifacts,'structure-vae.json'),'utf8'));
assert.equal(model.sampleCount, expected.sampleCount);
assert.ok(model.weights.w_mu.length > 0);
const matches = rankCadLatentModelSamples('拉伸 圆形草图 切除', model, {limit:3});
assert.equal(matches.length, 3);
assert.ok(matches.every(m => Number.isFinite(m.rankScore) && Number.isFinite(m.latentDistance)));
console.log(JSON.stringify({passed:true,format:model.format,sampleCount:model.sampleCount,query:'拉伸 圆形草图 切除',matchedIds:matches.map(m=>m.id),runtimeDir}, null, 2));
