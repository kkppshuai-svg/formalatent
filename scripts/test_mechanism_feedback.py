import unittest

try:
    import cadquery as cq
except ImportError:
    cq = None

if cq is not None:
    from verify_mechanism_examples import verify_candidates


@unittest.skipIf(cq is None, 'CadQuery is not installed')
class MechanismFeedbackTests(unittest.TestCase):
    def slider(self):
        locations = {'base': [0, 0, -2], 'slide': [0, 0, 1], 'decoy': [0, 12, 1]}
        return {
            'plan': {
                'parts': [{'id': key, 'pose': {'translate': xyz}} for key, xyz in locations.items()],
                'joints': [
                    {'id': 'ground', 'type': 'fixed', 'parent': 'world', 'child': 'base'},
                    {'id': 'drive', 'type': 'prismatic', 'parent': 'base', 'child': 'slide',
                     'origin': {'xyz': [0, 0, 0]}, 'axis': [1, 0, 0]},
                ],
            },
            'probe': {'jointId': 'drive', 'movingPart': 'slide', 'stationaryParts': ['base', 'decoy'],
                      'travel': 4, 'axis': [1, 0, 0], 'pivot': [0, 0, 0]},
        }

    @staticmethod
    def build(_part):
        return cq.Workplane('XY').box(2, 2, 2)

    def test_valid_replay_can_still_move_the_wrong_part(self):
        checks = verify_candidates(self.slider(), self.build)
        self.assertTrue(checks['slide']['motionPassed'])
        self.assertTrue(checks['decoy']['replayPassed'])
        self.assertFalse(checks['decoy']['motionPassed'])
        self.assertEqual(checks['decoy']['reason'], 'motion_goal_mismatch')

    def test_rest_clearance_does_not_imply_motion_clearance(self):
        example = self.slider()
        example['plan']['parts'].append({'id': 'stop', 'pose': {'translate': [5, 0, 1]}})
        example['probe']['stationaryParts'].append('stop')
        check = verify_candidates(example, self.build)['slide']
        self.assertTrue(check['replayPassed'])
        self.assertFalse(check['motionPassed'])
        self.assertEqual(check['reason'], 'motion_interference')
        self.assertAlmostEqual(check['collision']['volumeMm3'], 4)
