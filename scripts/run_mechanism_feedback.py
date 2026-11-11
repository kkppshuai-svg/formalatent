"""Compare joint prediction with and without feedback from CAD and motion checks."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from mechanism_examples import generate_examples
from train_aicad_graph_vae import (
    FrequencyBaseline, grouped_folds, masked_cases, save_checkpoint, train,
)
from verify_mechanism_examples import verify_candidates


GRAPH_EXPORT = '''
import { readFileSync } from "node:fs";
import { buildCadStructureGraph } from "./cad-structure-graph.js";
const plans = JSON.parse(readFileSync(0, "utf8"));
console.log(JSON.stringify(plans.map(buildCadStructureGraph)));
'''


def export_graphs(examples, ai_cad):
    process = subprocess.run(
        ['node', '--input-type=module', '--eval', GRAPH_EXPORT],
        input=json.dumps([example['plan'] for example in examples]).encode('utf-8'),
        cwd=ai_cad, capture_output=True, check=True,
    )
    return json.loads(process.stdout)


def verified_cases(examples, graphs, build_part):
    records, evidence, cases, skipped = [], {}, [], []
    seen = set()
    for example, graph in zip(examples, graphs, strict=True):
        row = {'id': example['id'], 'family': example['family'], 'structureGraph': graph}
        plan = example['plan']
        joint = next(joint for joint in plan['joints'] if joint['id'] == example['probe']['jointId'])
        part_nodes = {part['id']: f'part:{i}' for i, part in enumerate(plan['parts'])}
        source = 'world' if joint['parent'] == 'world' else part_nodes[joint['parent']]
        node_parts = {node: part for part, node in part_nodes.items()}
        matches = [
            case for case in masked_cases(row)
            if case.task == 'joint_destination' and case.context[-1] == joint['type']
            and case.node_ids[case.anchors[0]] == source
            and case.node_ids[case.target] == part_nodes[joint['child']]
        ]
        if len(matches) != 1:
            skipped.append({'id': row['id'], 'reason': 'missing_or_ambiguous_joint_query'})
            continue
        case = matches[0]
        # Fingerprint the entire observed query, excluding the withheld answer.
        # Two identical inputs cannot serve as separate train/test evidence.
        graph_key = json.dumps([case.nodes, sorted(case.edges), case.orders, case.anchors,
                                case.context, case.candidates], sort_keys=True)
        if graph_key in seen:
            skipped.append({'id': row['id'], 'reason': 'duplicate_masked_graph'})
            continue
        verification = verify_candidates(example, build_part)
        outcomes = {candidate: verification[node_parts[case.node_ids[candidate]]]
                    for candidate in case.candidates}
        if not passed(outcomes[case.target]):
            skipped.append({'id': row['id'], 'reason': 'reference_failed',
                            'reference': outcomes[case.target]})
            continue
        hard_negatives = tuple(candidate for candidate, result in outcomes.items()
                               if result['replayPassed'] and not result['motionPassed'])
        seen.add(graph_key)
        cases.append(replace(case, hard_negatives=hard_negatives))
        evidence[row['id']] = outcomes
        records.append({**example, 'structureGraph': graph, 'candidates': [
            {'part': node_parts[case.node_ids[candidate]], 'node': case.node_ids[candidate],
             'reference': candidate == case.target, **result}
            for candidate, result in outcomes.items()
        ]})
    return cases, evidence, records, skipped


def passed(result):
    return result['replayPassed'] and result['motionPassed']


def neural_ranks(model, cases):
    with torch.no_grad():
        for start in range(0, len(cases), 32):
            batch = cases[start:start + 32]
            output = model(batch, sample=False)
            for i, case in enumerate(batch):
                scores, labels = model.choices(case, i, output)
                if not torch.isfinite(scores).all():
                    raise ValueError(f'non-finite scores for {case.record}')
                yield [label for label, _ in sorted(zip(labels, scores.tolist()),
                                                   key=lambda entry: entry[1], reverse=True)]


def score(cases, ranks, evidence=None):
    counts = Counter()
    predictions = []
    for case, rank in zip(cases, ranks, strict=True):
        first = rank[0] if rank else None
        top_three = rank[:3]
        position = rank.index(case.target) + 1 if case.target in rank else None
        counts.update(n=1, top1=int(position == 1), top3=int(position is not None and position <= 3))
        prediction = {'id': case.record, 'family': case.family, 'type': case.context[-1],
                      'source': case.node_ids[case.anchors[0]], 'reference': case.node_ids[case.target],
                      'rankedNodes': [case.node_ids[i] for i in rank], 'referenceRank': position}
        if evidence is not None:
            checks = evidence[case.record]
            best_verified = next((candidate for candidate in top_three if passed(checks[candidate])), None)
            counts.update(top1Replay=int(first is not None and checks[first]['replayPassed']),
                          top1ReplayAndMotion=int(first is not None and passed(checks[first])),
                          top3ReplayAndMotion=int(best_verified is not None),
                          verifiedTop3Reference=int(best_verified == case.target))
            prediction['firstVerifiedInTop3'] = (
                case.node_ids[best_verified] if best_verified is not None else None
            )
        predictions.append(prediction)
    return counts, predictions


def compare(training, testing, evidence, args, seed):
    baseline = FrequencyBaseline(training)
    results = {'frequency': score(testing, map(baseline.rank_destinations, testing), evidence)}
    trained = {}
    for method, weight in [('vae', 0.0), ('vaeWithFeedback', args.feedback_weight)]:
        model = train(training, args.epochs, seed, beta=0.005, interleave=True, feedback_weight=weight)
        results[method] = score(testing, neural_ranks(model, testing), evidence)
        trained[method] = model
    return results, trained


def previous_experience(args):
    if args.replay_dataset is None:
        return [], [], None
    raw = args.replay_dataset.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    replay = []
    testing = []
    for row in rows:
        cases = masked_cases(row)
        if row['family'] == args.real_test_family:
            testing.extend(case for case in cases if case.task == 'joint_destination')
        else:
            replay.extend(cases)
    if args.real_test_family and not testing:
        raise ValueError('held-out real family has no eligible joint target examples')
    return replay, testing, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-cad', type=Path, default=Path('../ai-cad'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--replay-dataset', type=Path)
    parser.add_argument('--real-test-family')
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123])
    parser.add_argument('--feedback-weight', type=float, default=1.0)
    args = parser.parse_args()
    if args.variants < 1 or args.epochs < 1 or not 0 < args.feedback_weight < float('inf'):
        parser.error('variants, epochs, and feedback weight must be positive')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct')
    if args.real_test_family and not args.replay_dataset:
        parser.error('--real-test-family requires --replay-dataset')
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.ai_cad.resolve() / 'scripts'))
    from cadquery_build import build_manifest_part

    examples = list(generate_examples(variants=args.variants, seed=42))
    cases, evidence, records, skipped = verified_cases(
        examples, export_graphs(examples, args.ai_cad), build_manifest_part
    )
    families = sorted({case.family for case in cases})
    if len(families) < 2 or not any(case.hard_negatives for case in cases):
        raise ValueError('experiment needs multiple verified families and replayable wrong-motion candidates')
    replay, real_testing, replay_sha = previous_experience(args)
    if set(families).intersection(case.family for case in replay + real_testing):
        raise ValueError('generated family names must be distinct from local experience')
    dataset = ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records).encode('utf-8')
    dataset_sha = hashlib.sha256(dataset).hexdigest()
    all_checks = [check for outcomes in evidence.values() for check in outcomes.values()]
    report = {
        'format': 'ai-cad-mechanism-feedback-experiment-v1', 'generatedPlans': len(examples),
        'verifiedPlans': len(records), 'skippedPlans': skipped, 'families': dict(Counter(c.family for c in cases)),
        'candidateCounts': dict(Counter(len(case.candidates) for case in cases)),
        'wrongMotionCandidates': sum(len(case.hard_negatives) for case in cases),
        'candidateCheckReasons': dict(Counter(check['reason'] for check in all_checks)),
        'randomExpectedTop3': sum(min(1, 3 / len(case.candidates)) for case in cases) / len(cases),
        'realReplayCases': len(replay), 'realReplaySha256': replay_sha,
        'datasetSha256': dataset_sha, 'epochs': args.epochs, 'feedbackWeight': args.feedback_weight,
        'runs': [],
    }
    release_model = None
    for seed in args.seeds:
        run = {'seed': seed, 'folds': [], 'metrics': {method: Counter()
               for method in ('frequency', 'vae', 'vaeWithFeedback')}}
        for held_out, training, testing in grouped_folds(cases, len(families), seed):
            results, _ = compare(training + replay, testing, evidence, args, seed)
            fold = {'testFamily': held_out[0], 'trainPlans': len(training),
                    'trainFamilies': sorted({case.family for case in training + replay}), 'methods': {}}
            for method, (counts, predictions) in results.items():
                run['metrics'][method].update(counts)
                fold['methods'][method] = {'metrics': counts, 'predictions': predictions}
            run['folds'].append(fold)
        if real_testing or seed == args.seeds[0]:
            results, models = compare(cases + replay, real_testing, None, args, seed)
            if seed == args.seeds[0]:
                release_model = models['vaeWithFeedback']
            if real_testing:
                run['realTransfer'] = {method: {'metrics': counts, 'predictions': predictions}
                                       for method, (counts, predictions) in results.items()}
        report['runs'].append(run)
        print(json.dumps({'seed': seed, 'metrics': run['metrics']}, ensure_ascii=False), flush=True)
    report['realTestFamily'] = args.real_test_family
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / 'report.json'
    (args.out_dir / 'verified-plans.jsonl').write_bytes(dataset)
    save_checkpoint(release_model, args.out_dir / 'vae-with-feedback.pt', dataset_sha)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'verifiedPlans': len(records), 'families': len(families),
                      'report': str(report_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
