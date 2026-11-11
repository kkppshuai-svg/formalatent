import unittest

from deepcad_structure_graph import extract


def document():
    return {
        'entities': {
            'sketch': {'type': 'Sketch', 'profiles': {'left': {}, 'right': {}}},
            'cut1': {'type': 'ExtrudeFeature', 'operation': 'CutFeatureOperation',
                     'profiles': [{'sketch': 'sketch', 'profile': 'left'}]},
            'cut2': {'type': 'ExtrudeFeature', 'operation': 'CutFeatureOperation',
                     'profiles': [{'sketch': 'sketch', 'profile': 'right'}]},
        },
        'sequence': [{'type': 'Sketch', 'entity': 'sketch'},
                     {'type': 'ExtrudeFeature', 'entity': 'cut1'},
                     {'type': 'ExtrudeFeature', 'entity': 'cut2'}],
    }


class StructureGraphTests(unittest.TestCase):
    def test_extracts_profile_references_and_order_without_inventing_body_targets(self):
        result = extract(document())
        self.assertIn({'from': 'operation:1:cut1', 'to': 'profile:sketch:left', 'type': 'uses_profile'}, result['edges'])
        self.assertIn({'from': 'operation:2:cut2', 'to': 'profile:sketch:right', 'type': 'uses_profile'}, result['edges'])
        self.assertIn({'from': 'operation:2:cut2', 'to': 'operation:1:cut1', 'type': 'follows'}, result['edges'])
        self.assertFalse(any(edge['type'] == 'target' for edge in result['edges']))

    def test_rejects_missing_references(self):
        doc = document()
        doc['entities']['cut1']['profiles'][0]['profile'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'missing profile'):
            extract(doc)

    def test_rejects_unsupported_operations(self):
        doc = document()
        doc['sequence'].append({'type': 'FilletFeature', 'entity': 'fillet'})
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            extract(doc)


if __name__ == '__main__':
    unittest.main()
