import unittest
from unittest.mock import patch

try:
    import cadquery as cq
except ImportError:
    cq = None

if cq is not None:
    from import_deepcad_subset import replay_document_bodies


@unittest.skipIf(cq is None, "CadQuery is not installed")
class DocumentBodyTests(unittest.TestCase):
    def test_explicit_cut_affects_only_selected_body(self):
        entities = {
            "left": {"operation": "NewBodyFeatureOperation"},
            "right": {"operation": "NewBodyFeatureOperation"},
            "cut": {"operation": "CutFeatureOperation", "targetBody": "left"},
        }
        document = {
            "entities": entities,
            "sequence": [{"type": "ExtrudeFeature", "entity": key} for key in entities],
        }
        shapes = [
            cq.Solid.makeBox(4, 4, 4),
            cq.Solid.makeBox(4, 4, 4, cq.Vector(10, 0, 0)),
            cq.Solid.makeBox(2, 2, 4, cq.Vector(1, 1, 0)),
        ]
        with patch("import_deepcad_subset.extruded_shape", side_effect=shapes):
            bodies = replay_document_bodies(document)
        self.assertEqual(list(bodies), ["left", "right"])
        self.assertAlmostEqual(bodies["left"].Volume(), 48)
        self.assertAlmostEqual(bodies["right"].Volume(), 64)
        del entities["cut"]["targetBody"]
        with patch("import_deepcad_subset.extruded_shape", side_effect=shapes):
            with self.assertRaisesRegex(ValueError, "targetBody"):
                replay_document_bodies(document)
