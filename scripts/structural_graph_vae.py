"""Experimental graph VAE for profile-to-operation references.

The graph contains a fixed number of profile nodes and operation nodes. Each
operation has a separate variational representation for its profile reference.
"""
import argparse
import itertools
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ReferenceGraphVAE(nn.Module):
    def __init__(self, profiles=3, width=32, latent=4):
        super().__init__()
        self.profiles = profiles
        self.profile_embedding = nn.Embedding(profiles, width)
        self.operation_query = nn.Parameter(torch.randn(width) * 0.02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mu = nn.Linear(width, latent)
        self.logvar = nn.Linear(width, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, width), nn.Tanh(), nn.Linear(width, profiles))

    def encode(self, references, sample=True):
        batch, operations = references.shape
        profile_ids = torch.arange(self.profiles, device=references.device)
        nodes = self.profile_embedding(profile_ids).expand(batch * operations, -1, -1)
        query = self.operation_query.view(1, 1, -1).expand(batch * operations, -1, -1)
        # An operation can attend only to the profile referenced by its edge.
        mask = profile_ids.view(1, -1) != references.reshape(-1, 1)
        hidden, _ = self.attention(query, nodes, nodes, key_padding_mask=mask, need_weights=False)
        hidden = hidden.reshape(batch, operations, -1)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-8, 5)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)


def training_loss(model, base, source, slot, beta, intervention_weight):
    z, mu, logvar = model.encode(base)
    source_z, _, _ = model.encode(source)
    reconstruction = F.cross_entropy(model.decode(z).reshape(-1, model.profiles), base.reshape(-1))
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    selector = F.one_hot(slot, num_classes=base.shape[1]).bool().unsqueeze(-1)
    swapped = torch.where(selector, source_z, z)
    expected = torch.where(selector.squeeze(-1), source, base)
    intervention = F.cross_entropy(model.decode(swapped).reshape(-1, model.profiles), expected.reshape(-1))
    return reconstruction + beta * kl + intervention_weight * intervention


def score(model, records):
    model.eval()
    with torch.no_grad():
        z, _, _ = model.encode(records, sample=False)
        prediction = model.decode(z).argmax(-1)
        correct = (prediction == records).float().mean().item()
        exact = (prediction == records).all(-1).float().mean().item()
        source = records.flip(0)
        source_z, _, _ = model.encode(source, sample=False)
        metrics = []
        for slot in range(records.shape[1]):
            switched = z.clone()
            switched[:, slot] = source_z[:, slot]
            out = model.decode(switched).argmax(-1)
            expected = records.clone()
            expected[:, slot] = source[:, slot]
            changed = records[:, slot] != source[:, slot]
            if changed.any():
                metrics.append((out[changed] == expected[changed]).all(-1).float())
        intervention = torch.cat(metrics).mean().item() if metrics else None
    return {'referenceAccuracy': correct, 'graphExactAccuracy': exact,
            'interventionGraphAccuracy': intervention, 'samples': len(records)}


def verify_cad(references):
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError('CAD verification requires CadQuery') from exc
    plate = cq.Workplane('XY').box(50, 20, 4, centered=(True, True, False)).val()
    cutters = [cq.Solid.makeCylinder(2, 4, cq.Vector(x, 0, 0)) for x in (-12, 0, 12)]
    shape = plate
    for reference in references:
        shape = shape.cut(cutters[reference])
    expected = set(references)
    remaining = [float(shape.intersect(cutter).Volume()) for cutter in cutters]
    valid = bool(shape.isValid() and shape.Solids())
    for i, cutter in enumerate(cutters):
        expected_volume = 0 if i in expected else cutter.Volume()
        valid = valid and abs(remaining[i] - expected_volume) < 1e-5
    return {'valid': bool(valid), 'remainingVolumes': remaining}


def run(epochs, seed, output, intervention_weight, cad=False, profiles=3):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    combinations = list(itertools.product(range(profiles), repeat=2))
    # Hold out combinations, while retaining every individual reference in training.
    held_out = [(i, (i + 2) % profiles) for i in range(profiles)]
    train = torch.tensor([x for x in combinations if x not in held_out], dtype=torch.long)
    test = torch.tensor(held_out, dtype=torch.long)
    model = ReferenceGraphVAE(profiles=profiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        base = train[torch.randint(len(train), (48,), generator=generator)]
        source = train[torch.randint(len(train), (48,), generator=generator)]
        slot = torch.randint(2, (48,), generator=generator)
        optimizer.zero_grad()
        loss = training_loss(model, base, source, slot, beta=0.001,
                             intervention_weight=intervention_weight)
        loss.backward()
        optimizer.step()
    result = {'train': score(model, train), 'heldOutCombinations': score(model, test),
              'heldOut': held_out, 'epochs': epochs, 'seed': seed,
              'interventionWeight': intervention_weight}
    if cad:
        base = torch.tensor([[0, 1]])
        source = torch.tensor([[2, 0]])
        with torch.no_grad():
            base_z, _, _ = model.encode(base, sample=False)
            source_z, _, _ = model.encode(source, sample=False)
            before = model.decode(base_z).argmax(-1)[0].tolist()
            base_z[:, 0] = source_z[:, 0]
            after = model.decode(base_z).argmax(-1)[0].tolist()
        result['cadIntervention'] = {'baseReferences': before, 'changedReferences': after,
                                     'base': verify_cad(before), 'changed': verify_cad(after),
                                     'expectedEdit': before == [0, 1] and after == [2, 1]}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': 'formalatent-reference-graph-vae-v1',
                    'profiles': profiles, 'width': 32, 'latent': 4,
                    'state_dict': model.state_dict(), 'evaluation': result}, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--intervention-weight', type=float, default=1.0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cad', action='store_true')
    parser.add_argument('--profiles', type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.intervention_weight < 0 or not 3 <= args.profiles <= 16:
        parser.error('epochs must be positive, intervention weight nonnegative and profiles between 3 and 16')
    if args.cad and args.profiles != 3:
        parser.error('the built-in CAD demonstration has exactly three profiles')
    print(json.dumps(run(args.epochs, args.seed, args.out, args.intervention_weight, args.cad, args.profiles), indent=2))


if __name__ == '__main__':
    main()
