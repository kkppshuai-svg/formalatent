"""Reconstruct or interchange profile-reference latents in a trained graph VAE."""
import argparse
import copy
import json
from pathlib import Path

import torch

from structural_graph_vae import ReferenceGraphVAE, verify_cad
from deepcad_structure_graph import extract


def references(value, profiles):
    if not isinstance(value, list) or not value or any(type(i) is not int or i < 0 or i >= profiles for i in value):
        raise ValueError(f'references must be a nonempty list of integers from 0 to {profiles - 1}')
    return torch.tensor([value], dtype=torch.long)


def graph_references(graph, profiles):
    if graph.get('format') != 'formalatent-feature-graph-v1':
        raise ValueError('unsupported graph format')
    nodes = graph['nodes']
    profile_ids = [node['id'] for node in nodes if node.get('type') == 'profile']
    if len(profile_ids) != profiles or len(set(profile_ids)) != profiles:
        raise ValueError(f'graph must have exactly {profiles} unique profiles')
    operations = sorted((node for node in nodes if node.get('type') == 'extrude'), key=lambda x: x['order'])
    values = []
    for operation in operations:
        matches = [edge['to'] for edge in graph['edges']
                   if edge.get('from') == operation['id'] and edge.get('type') == 'uses_profile']
        if len(matches) != 1 or matches[0] not in profile_ids:
            raise ValueError('each operation must reference one known profile')
        values.append(profile_ids.index(matches[0]))
    return references(values, profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--cad', action='store_true')
    parser.add_argument('--document', type=Path, help='replay references in this original DeepCAD JSON')
    parser.add_argument('--document-out', type=Path, help='save the modified DeepCAD document')
    parser.add_argument('--step-out', type=Path, help='export the final replayed solid')
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'formalatent-reference-graph-vae-v1':
        raise ValueError('unsupported model format')
    model = ReferenceGraphVAE(checkpoint['profiles'], checkpoint['width'], checkpoint['latent'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if args.cad and model.profiles != 3:
        raise ValueError('the built-in CAD demonstration has exactly three profiles')
    doc = json.loads(args.input.read_text(encoding='utf-8'))
    if (args.step_out or args.document_out) and not args.document:
        raise ValueError('output paths require --document')
    base = (graph_references(doc['baseGraph'], model.profiles) if 'baseGraph' in doc
            else references(doc['base'], model.profiles))
    with torch.no_grad():
        z, _, _ = model.encode(base, sample=False)
        before = model.decode(z).argmax(-1)[0].tolist()
        result = {'base': before}
        if 'source' in doc or 'sourceGraph' in doc:
            if 'baseGraph' in doc and 'sourceGraph' in doc:
                def ids(graph, kind):
                    return [node['id'] for node in graph['nodes'] if node.get('type') == kind]
                for kind in ('profile', 'extrude'):
                    if ids(doc['baseGraph'], kind) != ids(doc['sourceGraph'], kind):
                        raise ValueError(f'base and source {kind} identities and order must match')
            source = (graph_references(doc['sourceGraph'], model.profiles) if 'sourceGraph' in doc
                      else references(doc['source'], model.profiles))
            slot = doc['slot']
            if source.shape != base.shape or type(slot) is not int or not 0 <= slot < base.shape[1]:
                raise ValueError('source must have the same operation count and slot must be in range')
            source_z, _, _ = model.encode(source, sample=False)
            z[:, slot] = source_z[:, slot]
            result['intervened'] = model.decode(z).argmax(-1)[0].tolist()
        if args.cad:
            result['cad'] = {key: verify_cad(value) for key, value in result.items()}
    if args.document:
        if 'baseGraph' not in doc:
            raise ValueError('--document requires baseGraph in the input')
        original = json.loads(args.document.read_text(encoding='utf-8'))
        graph = extract(original)
        if graph != doc['baseGraph']:
            raise ValueError('baseGraph must match the original document')
        import cadquery as cq
        from import_deepcad_subset import replay_document_bodies
        profile_map = {}
        for sketch_id, entity in original['entities'].items():
            if entity.get('type') == 'Sketch':
                for profile_id in entity.get('profiles', {}):
                    profile_map[f'profile:{sketch_id}:{profile_id}'] = {'sketch': sketch_id, 'profile': profile_id}
        profile_ids = [node['id'] for node in graph['nodes'] if node['type'] == 'profile']
        operations = sorted((node for node in graph['nodes'] if node['type'] == 'extrude'), key=lambda node: node['order'])
        output = copy.deepcopy(original)
        changes = {}
        for node, reference in zip(operations, result.get('intervened', result['base'])):
            entity_id = output['sequence'][node['order']]['entity']
            profiles = output['entities'][entity_id]['profiles']
            old = {key: profiles[0][key] for key in ('sketch', 'profile')}
            new = profile_map[profile_ids[reference]]
            if old != new:
                changes[entity_id] = {'feature': entity_id, 'from': old, 'to': new}
                output['entities'][entity_id]['profiles'] = [new]
        original_trace, edited_trace = [], []
        original_bodies = replay_document_bodies(original, trace=original_trace)
        bodies = replay_document_bodies(output, trace=edited_trace)
        result['structureExplanation'] = explain_replay(
            changes, original_trace, edited_trace, original_bodies, bodies
        )
        shape = cq.Compound.makeCompound(list(bodies.values()))
        result['documentReplay'] = {'valid': bool(shape.isValid() and shape.Solids() and shape.Volume() > 0),
                                    'volume': float(shape.Volume()), 'solidCount': len(shape.Solids()),
                                    'bodyVolumes': {key: float(body.Volume()) for key, body in bodies.items()}}
        if args.document_out:
            if not result['documentReplay']['valid']:
                raise ValueError('document replay produced an invalid or empty solid')
            args.document_out.parent.mkdir(parents=True, exist_ok=True)
            args.document_out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            result['documentReplay']['documentPath'] = str(args.document_out)
        if args.step_out:
            if not result['documentReplay']['valid']:
                raise ValueError('document replay produced an invalid or empty solid')
            args.step_out.parent.mkdir(parents=True, exist_ok=True)
            cq.exporters.export(shape, str(args.step_out))
            result['documentReplay']['stepPath'] = str(args.step_out)
    print(json.dumps(result, indent=2))


def explain_replay(changes, original_trace, edited_trace, original_bodies, bodies):
    affected = set()
    later_operations = []
    for original, edited in zip(original_trace, edited_trace, strict=True):
        feature, body = edited['feature'], edited['body']
        if body in affected:
            later_operations.append({
                'feature': feature, 'body': body, 'operation': edited['operation'],
                'originalVolumeAfter': original['volumeAfter'],
                'editedVolumeAfter': edited['volumeAfter'],
            })
        if feature in changes:
            affected.add(body)
            changes[feature]['body'] = body
    geometry = []
    for key in sorted(affected):
        original, edited = original_bodies[key], bodies[key]
        geometry.append({
            'body': key,
            'removedVolume': float(original.cut(edited).Volume()),
            'addedVolume': float(edited.cut(original).Volume()),
        })
    return {'referenceChanges': list(changes.values()), 'laterOperationsOnEditedBodies': later_operations,
            'finalBodyGeometryChanges': geometry}


if __name__ == '__main__':
    main()
