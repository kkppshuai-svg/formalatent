import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from structural_graph_cli import graph_references
    from structural_graph_vae import ReferenceGraphVAE, training_loss


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class GraphVaeTests(unittest.TestCase):
    def test_swapping_one_operation_preserves_other_decoded_values(self):
        torch.manual_seed(5)
        model = ReferenceGraphVAE()
        base = torch.tensor([[0, 1]])
        source = torch.tensor([[2, 0]])
        z, _, _ = model.encode(base, sample=False)
        source_z, _, _ = model.encode(source, sample=False)
        before = model.decode(z)
        z[:, 0] = source_z[:, 0]
        after = model.decode(z)
        self.assertTrue(torch.equal(before[:, 1], after[:, 1]))

    def test_intervention_loss_backpropagates(self):
        model = ReferenceGraphVAE()
        loss = training_loss(model, torch.tensor([[0, 1], [2, 0]]),
                             torch.tensor([[2, 1], [0, 0]]), torch.tensor([0, 0]),
                             beta=0.001, intervention_weight=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.mu.weight.grad.abs().sum()), 0)

    def test_graph_reference_adapter_rejects_multi_profile_operations(self):
        graph = {'format': 'formalatent-feature-graph-v1',
                 'nodes': [{'id': str(i), 'type': 'profile'} for i in range(3)] +
                          [{'id': 'op', 'type': 'extrude', 'order': 0}],
                 'edges': [{'from': 'op', 'to': '0', 'type': 'uses_profile'},
                           {'from': 'op', 'to': '1', 'type': 'uses_profile'}]}
        with self.assertRaisesRegex(ValueError, 'one known profile'):
            graph_references(graph, 3)


if __name__ == '__main__':
    unittest.main()
