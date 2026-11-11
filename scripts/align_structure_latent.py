"""Learn a one-dimensional interchange direction for an existing structure VAE."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from structure_vae import StructureVAE, tokens


def load_pairs(path, model, token):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('pair file is empty')
    base = [row['base'] for row in rows]
    source = [row['source'] for row in rows]
    base_z = torch.tensor(model.encode(base), dtype=torch.float32)
    source_z = torch.tensor(model.encode(source), dtype=torch.float32)
    target = torch.tensor([float(token in tokens(item)) for item in source])
    if not torch.isfinite(base_z).all() or not torch.isfinite(source_z).all():
        raise ValueError('non-finite latent')
    return base_z, source_z, target


def logits(z, weights):
    return torch.tanh(z @ weights['w2'] + weights['b2']) @ weights['w_out'] + weights['b_out']


def evaluate(direction, base, source, target, index, weights, token_count):
    with torch.no_grad():
        direction = F.normalize(direction, dim=0)
        swapped = base + ((source - base) @ direction).unsqueeze(1) * direction
        out = logits(swapped, weights)
        baseline = logits(base, weights)
        changed = (torch.sigmoid(baseline[:, index]) >= 0.5) != target.bool()
        other = torch.ones(token_count, dtype=torch.bool)
        other[index] = False
        collateral = (torch.sigmoid(out[:, :token_count]) - torch.sigmoid(baseline[:, :token_count])).abs()
        return {'pairs': len(base), 'differentTargetPairs': int(changed.sum()),
                'targetAccuracy': float(((out[:, index] >= 0) == target.bool()).float().mean()),
                'changedTargetAccuracy': float(((out[changed, index] >= 0) == target[changed].bool()).float().mean()) if changed.any() else None,
                'meanOtherTokenProbabilityChange': float(collateral[:, other].mean()) if other.any() else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--train-pairs', required=True, type=Path)
    parser.add_argument('--test-pairs', required=True, type=Path)
    parser.add_argument('--token', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    model = StructureVAE(json.loads(args.model.read_text(encoding='utf-8')))
    if args.token not in model.vocabulary:
        raise ValueError('token is not in model vocabulary')
    index = model.vocabulary.index(args.token)
    base, source, target = load_pairs(args.train_pairs, model, args.token)
    test_base, test_source, test_target = load_pairs(args.test_pairs, model, args.token)
    weights = {name: torch.tensor(value, dtype=torch.float32) for name, value in model.params.items()}
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    direction = torch.nn.Parameter(torch.randn(model.model['latentDim']))
    optimizer = torch.optim.Adam([direction], lr=0.03)
    reference = logits(base, weights).detach()
    mask = torch.ones(len(model.vocabulary), dtype=torch.bool)
    mask[index] = False
    for _ in range(args.steps):
        optimizer.zero_grad()
        unit = F.normalize(direction, dim=0)
        swapped = base + ((source - base) @ unit).unsqueeze(1) * unit
        out = logits(swapped, weights)
        loss = F.binary_cross_entropy_with_logits(out[:, index], target)
        if mask.any():
            loss = loss + F.mse_loss(torch.sigmoid(out[:, :len(mask)][:, mask]),
                                     torch.sigmoid(reference[:, :len(mask)][:, mask]))
        loss.backward()
        optimizer.step()
    learned = F.normalize(direction.detach(), dim=0)
    random_direction = F.normalize(torch.randn_like(learned), dim=0)
    result = {'format': 'formalatent-latent-alignment-v1', 'token': args.token,
              'direction': learned.tolist(),
              'train': evaluate(learned, base, source, target, index, weights, len(mask)),
              'test': evaluate(learned, test_base, test_source, test_target, index, weights, len(mask)),
              'randomDirectionTest': evaluate(random_direction, test_base, test_source, test_target, index, weights, len(mask))}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
