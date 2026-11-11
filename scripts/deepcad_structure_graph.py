"""Extract explicit feature and profile references from a DeepCAD document."""
import argparse
import json
from pathlib import Path


def extract(document):
    entities = document['entities']
    nodes, edges = [], []
    sketches = {}
    for sketch_id, sketch in entities.items():
        if sketch.get('type') != 'Sketch':
            continue
        sketch_node = f'sketch:{sketch_id}'
        sketches[sketch_id] = sketch_node
        nodes.append({'id': sketch_node, 'type': 'sketch'})
        for profile_id in sketch.get('profiles', {}):
            profile_node = f'profile:{sketch_id}:{profile_id}'
            nodes.append({'id': profile_node, 'type': 'profile'})
            edges.append({'from': profile_node, 'to': sketch_node, 'type': 'belongs_to'})
    previous = None
    for index, item in enumerate(document['sequence']):
        if item.get('type') == 'Sketch':
            if item['entity'] not in sketches:
                raise ValueError('sequence references a missing sketch')
            continue
        if item.get('type') != 'ExtrudeFeature':
            raise ValueError(f'unsupported sequence operation: {item.get("type")}')
        entity_id = item['entity']
        entity = entities[entity_id]
        if entity.get('type') != 'ExtrudeFeature':
            raise ValueError('sequence entity type mismatch')
        operation_id = f'operation:{index}:{entity_id}'
        nodes.append({'id': operation_id, 'type': 'extrude', 'operation': entity['operation'], 'order': index})
        refs = entity.get('profiles', [])
        if not refs:
            raise ValueError('extrude has no profile references')
        for ref in refs:
            sketch_id, profile_id = ref['sketch'], ref['profile']
            if sketch_id not in sketches or profile_id not in entities[sketch_id].get('profiles', {}):
                raise ValueError('extrude references a missing profile')
            edges.append({'from': operation_id, 'to': f'profile:{sketch_id}:{profile_id}', 'type': 'uses_profile'})
        if previous is not None:
            # Sequence order is observable. It does not establish a Boolean target.
            edges.append({'from': operation_id, 'to': previous, 'type': 'follows'})
        previous = operation_id
    return {'format': 'formalatent-feature-graph-v1', 'nodes': nodes, 'edges': edges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    graph = extract(json.loads(args.document.read_text(encoding='utf-8')))
    text = json.dumps(graph, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
