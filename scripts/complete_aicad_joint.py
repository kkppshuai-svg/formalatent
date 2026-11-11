"""Complete one AI-CAD joint using VAE ranking and CAD/motion checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

from run_mechanism_feedback import export_graphs, neural_ranks, passed
from train_aicad_graph_vae import joint_completion_case, load_checkpoint
from verify_mechanism_examples import verify_candidates


def complete_joint(example, model, ai_cad, build_part):
    plan = example['plan']
    probe = example['probe']
    matches = [joint for joint in plan['joints'] if joint['id'] == probe['jointId']]
    if len(matches) != 1:
        raise ValueError('probe must identify exactly one joint')
    joint = matches[0]
    joint_type = joint['type']
    if joint_type not in ('revolute', 'prismatic'):
        raise ValueError('completion requires a revolute or prismatic joint')
    if 'joint_destination' not in model.tasks or joint_type not in model.edge_queries:
        raise ValueError(f'checkpoint does not cover {joint_type} joint completion')
    part_nodes = {part['id']: f'part:{i}' for i, part in enumerate(plan['parts'])}
    parent = joint['parent']
    if parent != 'world' and parent not in part_nodes:
        raise ValueError('joint parent must identify an existing part or world')
    parent_node = 'world' if parent == 'world' else part_nodes[parent]
    node_parts = {node: part for part, node in part_nodes.items()}
    observed = {**plan, 'joints': [item for item in plan['joints'] if item is not joint]}
    graph = export_graphs([{'plan': observed}], ai_cad)[0]
    case = joint_completion_case(graph, parent_node, joint_type, probe['jointId'])
    ranked = next(neural_ranks(model, [case]))[:3] if case.candidates else []
    candidates = [node_parts[case.node_ids[index]] for index in ranked]
    checks = verify_candidates(example, build_part, candidate_ids=candidates) if candidates else {}
    selected = next((part for part in candidates if passed(checks[part])), None)
    result = {
        'status': 'completed' if selected is not None else 'no_pass',
        'jointId': probe['jointId'], 'selectedChild': selected,
        'rankedCandidates': [{'part': part, **checks[part]} for part in candidates],
    }
    if selected is None:
        return result, None
    completed = deepcopy(plan)
    for item in completed['joints']:
        if item['id'] == probe['jointId']:
            item['child'] = selected
    return result, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-cad', type=Path, default=Path(__file__).resolve().parents[2] / 'ai-cad')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--plan-out', type=Path, required=True)
    args = parser.parse_args()
    if args.plan_out.resolve() in (args.input.resolve(), args.model.resolve()):
        parser.error('--plan-out must differ from the input and checkpoint')
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.ai_cad.resolve() / 'scripts'))
    from cadquery_build import build_manifest_part

    example = json.loads(args.input.read_text(encoding='utf-8-sig'))
    model = load_checkpoint(args.model)
    result, plan = complete_joint(example, model, args.ai_cad, build_manifest_part)
    if plan is not None:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        result['planOutput'] = str(args.plan_out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if plan is not None else 2


if __name__ == '__main__':
    sys.exit(main())
