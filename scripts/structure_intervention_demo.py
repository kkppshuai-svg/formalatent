"""Train and test a small VAE that learns a cut-to-profile relation.

The demonstration uses the existing structure VAE and a deliberately limited
two-profile vocabulary. Run with --cad to additionally validate the two
decoded alternatives with CadQuery.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from structure_vae import StructureVAE, train


LEFT = 'reference:cut:profile-left'
RIGHT = 'reference:cut:profile-right'
COMMON = ['feature:plate', 'feature:cut', 'profile:left', 'profile:right']


def sample(target, identifier):
    if target not in ('left', 'right'):
        raise ValueError('target must be left or right')
    return {
        'id': identifier,
        'tokens': COMMON + [LEFT if target == 'left' else RIGHT],
        'structure': {'partCount': 1, 'jointCount': 0},
    }


def decoded_target(model, latent):
    probs = model.decode(np.asarray(latent, dtype=float).reshape(1, -1))['tokenProbabilities'][0]
    values = {token: float(probs[model.vocabulary.index(token)]) for token in (LEFT, RIGHT)}
    return ('left' if values[LEFT] > values[RIGHT] else 'right'), values


def graph(target):
    return {
        'nodes': [
            {'id': 'plate', 'type': 'body'},
            {'id': 'left', 'type': 'circle_profile'},
            {'id': 'right', 'type': 'circle_profile'},
            {'id': 'cut', 'type': 'cut'},
        ],
        'edges': [
            {'from': 'cut', 'to': 'plate', 'type': 'target'},
            {'from': 'cut', 'to': target, 'type': 'profile'},
        ],
    }


def verify_cad(target):
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError('CAD 验证需要安装 requirements-cad.txt') from exc
    # The two possible cuts are fully inside the plate and disjoint.
    plate = cq.Workplane('XY').box(40, 20, 4, centered=(True, True, False)).val()
    tools = {name: cq.Solid.makeCylinder(2, 4, cq.Vector(x, 0, 0))
             for name, x in [('left', -10), ('right', 10)]}
    result = plate.cut(tools[target])
    remaining = {name: float(result.intersect(tool).Volume()) for name, tool in tools.items()}
    removed = float(plate.Volume() - result.Volume())
    expected = float(tools[target].Volume())
    valid = bool(result.isValid() and result.Solids() and abs(removed - expected) < 1e-5
                 and remaining[target] < 1e-5
                 and abs(remaining['right' if target == 'left' else 'left'] - expected) < 1e-5)
    return {'valid': valid, 'removedVolume': removed, 'expectedVolume': expected,
            'remainingVolumeAtProfiles': remaining}


def run(seed, epochs, cad):
    training = [sample(side, f'{side}-{i}') for i in range(40) for side in ('left', 'right')]
    payload = train(training, latent_dim=1, hidden_dim=16, epochs=epochs,
                    beta=0.001, seed=seed, patience=epochs,
                    count_mask_probability=0, batch_size=16)
    model = StructureVAE(payload)
    left, right = sample('left', 'base'), sample('right', 'source')
    latents = model.encode([left, right])
    original, original_probs = decoded_target(model, latents[0])
    source, source_probs = decoded_target(model, latents[1])
    # This demonstration has exactly one latent coordinate. The intervention
    # replaces it with the source's coordinate and decodes the resulting value.
    intervened = latents[0].copy()
    intervened[0] = latents[1, 0]
    changed, changed_probs = decoded_target(model, intervened)
    passed = original == 'left' and source == 'right' and changed == 'right'
    result = {
        'experiment': 'two-profile-cut-reference',
        'modelFormat': payload['format'],
        'scope': 'synthetic two-profile examples; a single latent coordinate',
        'latent': {'base': latents[0].tolist(), 'source': latents[1].tolist(),
                   'intervened': intervened.tolist()},
        'base': {'decodedTarget': original, 'probabilities': original_probs, 'graph': graph(original)},
        'source': {'decodedTarget': source, 'probabilities': source_probs},
        'intervened': {'decodedTarget': changed, 'probabilities': changed_probs, 'graph': graph(changed)},
        'passed': passed,
        'training': {'seed': seed, 'bestEpoch': payload['training']['bestEpoch'],
                     'sampleCount': len(training)},
        'cad': None,
    }
    if cad:
        result['cad'] = {'base': verify_cad(original), 'intervened': verify_cad(changed)}
        result['passed'] = passed and all(item['valid'] for item in result['cad'].values())
    return payload, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--cad', action='store_true')
    parser.add_argument('--model-out', type=Path, help='save the trained VAE JSON')
    args = parser.parse_args()
    payload, result = run(args.seed, args.epochs, args.cad)
    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.model_out.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
