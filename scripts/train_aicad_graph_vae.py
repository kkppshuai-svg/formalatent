"""Train and evaluate a masked VAE on AI-CAD assembly structure graphs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Case:
    record: str
    family: str
    task: str
    nodes: list
    edges: list
    anchors: tuple
    target: str | int
    context: tuple
    orders: list
    node_ids: list
    candidates: tuple = ()


def masked_cases(row):
    graph = row['structureGraph']
    if graph['format'] != 'ai-cad-structure-graph-v1':
        raise ValueError('unsupported AI-CAD graph format')
    nodes = graph['nodes']
    by_id = {node['id']: i for i, node in enumerate(nodes)}
    if len(nodes) != len(by_id):
        raise ValueError('duplicate graph node')
    roles = [node.get('role', '') for node in nodes]
    orders = [node.get('order', -1) for node in nodes]
    node_ids = [node['id'] for node in nodes]
    symbols = [(node['kind'], node.get('type', ''), roles[i])
               for i, node in enumerate(nodes)]
    edges = list(dict.fromkeys((by_id[e['from']], by_id[e['to']], e['kind'], e.get('type', ''))
                               for e in graph['edges']))
    owners = {destination: source for source, destination, kind, _ in edges if kind == 'contains'}
    parts = tuple(i for i, node in enumerate(nodes) if node['kind'] == 'part')
    cases = []

    def add(task, observed_nodes, observed_edges, anchors, target, context, candidates=()):
        cases.append(Case(row['id'], row['family'], task, observed_nodes, observed_edges,
                          anchors, target, context, orders, node_ids, candidates))

    for i, (kind, label, _) in enumerate(symbols):
        if kind in ('primitive', 'feature') and label and label != 'unknown' and i in owners:
            parent = owners[i]
            observed = symbols.copy()
            observed[i] = (kind, '', '')
            add(kind, observed, edges, (i, parent), label, (kind, roles[parent]))
    destinations = defaultdict(set)
    for source, destination, kind, label in edges:
        destinations[source, kind, label].add(destination)
    for i, (source, destination, kind, label) in enumerate(edges):
        already_connected = destinations[source, kind, label] - {destination}
        if kind == 'depends_on' and source in owners:
            # The observed tree order constrains the available prior features.
            candidates = tuple(node for node, owner in owners.items()
                               if owner == owners[source] and orders[node] < orders[source]
                               and node not in already_connected)
            if destination in candidates and len(candidates) >= 2:
                task = 'dependency_destination'
                add(task, symbols, edges[:i] + edges[i+1:], (source, source), destination,
                    (task, symbols[source][1]), candidates)
            continue
        if kind not in ('relation', 'joint') or not label:
            continue
        context = (kind, roles[source], roles[destination])
        observed = edges.copy()
        observed[i] = (source, destination, kind, '')
        add(kind, symbols, observed, (source, destination), label, context)
        candidates = tuple(p for p in parts if p != source and p not in already_connected)
        if destination in candidates and len(candidates) >= 2:
            task = f'{kind}_destination'
            add(task, symbols, edges[:i] + edges[i+1:], (source, source), destination,
                (task, roles[source], label), candidates)
    return cases


def dictionary(items):
    return {item: i for i, item in enumerate(sorted(set(items)), start=1)}


def vocabularies(cases):
    return {
        'kinds': dictionary(kind for case in cases for kind, _, _ in case.nodes),
        'types': dictionary((kind, label) for case in cases for kind, label, _ in case.nodes),
        'roles': dictionary(role for case in cases for _, _, role in case.nodes),
        'edges': dictionary((kind, label) for case in cases for _, _, kind, label in case.edges),
        'tasks': dictionary(case.task for case in cases),
        'labels': dictionary((case.task, case.target) for case in cases if not case.candidates),
        'edge_queries': dictionary(case.context[-1] for case in cases if case.candidates),
    }


class GraphVAE(nn.Module):
    def __init__(self, vocabulary, width=48, latent=16):
        super().__init__()
        self.vocabulary = vocabulary
        self.width = width
        self.latent = latent
        self.kinds = vocabulary['kinds']
        self.types = vocabulary['types']
        self.roles = vocabulary['roles']
        self.edges = vocabulary['edges']
        self.tasks = vocabulary['tasks']
        self.labels = vocabulary['labels']
        self.edge_queries = vocabulary['edge_queries']
        self.kind_embedding = nn.Embedding(len(self.kinds) + 1, width, padding_idx=0)
        self.type_embedding = nn.Embedding(len(self.types) + 1, width, padding_idx=0)
        self.role_embedding = nn.Embedding(len(self.roles) + 1, width, padding_idx=0)
        self.edge_embedding = nn.Embedding(len(self.edges) + 1, width, padding_idx=0)
        self.task_embedding = nn.Embedding(len(self.tasks) + 1, width, padding_idx=0)
        self.query_embedding = nn.Embedding(len(self.edge_queries) + 1, width, padding_idx=0)
        self.order_embedding = nn.Linear(1, width, bias=False)
        self.messages = nn.ModuleList(nn.Linear(width * 3, width) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.messages)
        self.posterior = nn.Linear(width * 5, latent * 2)
        self.decoder = nn.Sequential(nn.Linear(latent, width), nn.GELU(),
                                     nn.Linear(width, len(self.labels) + 1))
        self.pointer = nn.Sequential(nn.Linear(latent + width, width), nn.GELU(), nn.Linear(width, 1))
        self.label_indices = defaultdict(list)
        for (task, _), index in self.labels.items():
            self.label_indices[task].append(index)
        self.index_labels = {index: label for (_, label), index in self.labels.items()}

    def batch(self, cases):
        node_ids, edge_ids, groups, anchors, tasks, queries, offsets, orders = [], [], [], [], [], [], [], []
        for i, case in enumerate(cases):
            offset = len(node_ids)
            offsets.append(offset)
            node_ids.extend((self.kinds.get(kind, 0), self.types.get((kind, label), 0),
                             self.roles.get(role, 0)) for kind, label, role in case.nodes)
            groups.extend([i] * len(case.nodes))
            orders.extend(case.orders)
            edge_ids.extend((a + offset, b + offset, self.edges.get((kind, label), 0))
                            for a, b, kind, label in case.edges)
            anchors.append(tuple(j + offset for j in case.anchors))
            tasks.append(self.tasks.get(case.task, 0))
            queries.append(self.edge_queries.get(case.context[-1], 0) if case.candidates else 0)
        return (torch.tensor(node_ids), torch.tensor(edge_ids, dtype=torch.long).reshape(-1, 3),
                torch.tensor(groups), torch.tensor(anchors), torch.tensor(tasks), torch.tensor(queries),
                offsets, torch.tensor(orders, dtype=torch.float32).reshape(-1, 1))

    def forward(self, cases, sample):
        nodes, edges, groups, anchors, tasks, queries, offsets, orders = self.batch(cases)
        kinds, types, roles = nodes.T
        hidden = self.kind_embedding(kinds) + self.type_embedding(types) + self.role_embedding(roles)
        hidden = hidden + self.order_embedding(torch.log1p(orders + 1))
        source, destination, edge_ids = edges.T
        edge_vectors = self.edge_embedding(edge_ids)
        for layer, norm in zip(self.messages, self.norms):
            incoming = torch.zeros_like(hidden).index_add(0, destination, hidden[source] + edge_vectors)
            outgoing = torch.zeros_like(hidden).index_add(0, source, hidden[destination] + edge_vectors)
            hidden = norm(hidden + F.gelu(layer(torch.cat((hidden, incoming, outgoing), dim=1))))
        pooled = torch.zeros(len(cases), hidden.shape[1]).index_add(0, groups, hidden)
        pooled = pooled / torch.bincount(groups).unsqueeze(1)
        context = torch.cat((pooled, hidden[anchors[:, 0]], hidden[anchors[:, 1]],
                             self.task_embedding(tasks), self.query_embedding(queries)), dim=1)
        mean, logvar = self.posterior(context).chunk(2, dim=1)
        logvar = logvar.clamp(-8, 5)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return self.decoder(z), hidden, z, offsets, -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()

    def choices(self, case, index, output):
        logits, hidden, z, offsets, _ = output
        if case.task not in self.tasks:
            return logits.new_empty(0), ()
        if case.candidates:
            candidates = hidden[[offsets[index] + i for i in case.candidates]]
            query = z[index].expand(len(candidates), -1)
            return self.pointer(torch.cat((query, candidates), dim=1)).squeeze(1), case.candidates
        indices = self.label_indices.get(case.task, [])
        return logits[index, indices], tuple(self.index_labels[i] for i in indices)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'ai-cad-masked-graph-vae-v1':
        raise ValueError('unsupported AI-CAD graph VAE checkpoint')
    model = GraphVAE(checkpoint['vocabularies'], checkpoint['width'], checkpoint['latent'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


class FrequencyBaseline:
    def __init__(self, cases):
        self.contexts = defaultdict(Counter)
        self.tasks = defaultdict(Counter)
        for case in cases:
            label = self.destination_label(case, case.target) if case.candidates else case.target
            self.contexts[case.context][label] += 1
            self.tasks[case.task][label] += 1

    def predict(self, case):
        context, fallback = self.contexts[case.context], self.tasks[case.task]
        if not fallback:
            return None
        if case.candidates:
            return max(case.candidates, key=lambda i: (
                context[self.destination_label(case, i)], fallback[self.destination_label(case, i)]
            ))
        return (context or fallback).most_common(1)[0][0] if (context or fallback) else None

    @staticmethod
    def destination_label(case, index):
        return case.nodes[index][1 if case.task == 'dependency_destination' else 2]


def train(cases, epochs, seed, beta):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GraphVAE(vocabularies(cases))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    order = list(range(len(cases)))
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            batch = [cases[i] for i in order[start:start + 32]]
            output = model(batch, sample=True)
            losses = []
            for i, case in enumerate(batch):
                logits, labels = model.choices(case, i, output)
                losses.append(-F.log_softmax(logits, dim=0)[labels.index(case.target)])
            loss = torch.stack(losses).mean() + beta * output[-1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def evaluate(model, training, testing):
    baseline = FrequencyBaseline(training)
    counts = defaultdict(Counter)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(testing), 32):
            batch = testing[start:start + 32]
            output = model(batch, sample=False)
            for i, case in enumerate(batch):
                logits, labels = model.choices(case, i, output)
                prediction = labels[int(logits.argmax())] if labels else None
                reference = baseline.predict(case)
                supported = case.target in labels
                for key in (case.task, 'all'):
                    counts[key].update(n=1, neural=int(prediction == case.target),
                                       baseline=int(reference == case.target), supported=int(supported),
                                       neuralOnSupported=int(supported and prediction == case.target),
                                       baselineOnSupported=int(supported and reference == case.target))
                def display(value):
                    return case.node_ids[value] if case.candidates and value is not None else value
                predictions.append({'record': case.record, 'family': case.family, 'task': case.task,
                                    'anchors': [case.node_ids[j] for j in case.anchors],
                                    'target': display(case.target), 'prediction': display(prediction),
                                    'baseline': display(reference), 'targetSupported': supported})
    return counts, predictions


def grouped_folds(cases, folds, seed):
    families = sorted({case.family for case in cases})
    random.Random(seed).shuffle(families)
    groups = [families[i::folds] for i in range(folds)]
    return [(group, [case for case in cases if case.family not in group],
             [case for case in cases if case.family in group]) for group in groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dataset = args.dataset.read_bytes()
    rows = [json.loads(line) for line in dataset.splitlines() if line.strip()]
    if len({row['id'] for row in rows}) != len(rows):
        parser.error('dataset record IDs must be unique')
    cases = [case for row in rows for case in masked_cases(row)]
    families = sorted({case.family for case in cases})
    if args.epochs < 1 or not 2 <= args.folds <= len(families):
        parser.error('epochs must be positive and folds must be between 2 and the number of usable families')
    beta = 0.005
    total = defaultdict(Counter)
    results = []
    for group, training, testing in grouped_folds(cases, args.folds, args.seed):
        model = train(training, args.epochs, args.seed, beta)
        counts, predictions = evaluate(model, training, testing)
        for key, value in counts.items():
            total[key].update(value)
        results.append({'testFamilies': group, 'trainRecords': len({case.record for case in training}),
                        'testRecords': len({case.record for case in testing}), 'metrics': counts,
                        'trainTasks': dict(Counter(case.task for case in training)),
                        'predictions': predictions})
    model = train(cases, args.epochs, args.seed, beta)
    report = {'datasetSha256': hashlib.sha256(dataset).hexdigest(), 'records': len(rows),
              'usableRecords': len({case.record for case in cases}), 'families': families, 'epochs': args.epochs,
              'seed': args.seed, 'beta': beta, 'baseline': 'contextual label or destination-role/type frequency',
              'outOfVocabularyLabelsCountAsIncorrect': True, 'metrics': total, 'folds': results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': 'ai-cad-masked-graph-vae-v1', 'state_dict': model.state_dict(),
                'vocabularies': model.vocabulary, 'width': model.width, 'latent': model.latent,
                'datasetSha256': report['datasetSha256']}, args.model)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'records': report['records'], 'families': len(families), 'metrics': total}, indent=2))


if __name__ == '__main__':
    main()
