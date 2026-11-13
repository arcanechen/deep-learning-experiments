import unittest

import torch
from torch import nn

from controlled_cnn import ControlledCNN, expected_parameters
from prepare_ablation_plan import build_plan


class ControlledCNNTests(unittest.TestCase):
    def test_all_planned_configs_dimensions_parameters_and_gradients(self):
        torch.set_num_threads(2)
        plan = build_plan()
        self.assertEqual(plan['unique_configurations'], 11)
        self.assertEqual(plan['planned_runs'], 33)
        for config in plan['configurations']:
            with self.subTest(config=config['id']):
                depth, width = config['depth_per_stage'], config['width']
                model = ControlledCNN(depth, width)
                self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in model.modules()), 3 * depth)
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected_parameters(depth, width))
                output = model(torch.randn(2, 1, 32, 32))
                self.assertEqual(tuple(output.shape), (2, 10))
                nn.CrossEntropyLoss()(output, torch.tensor([0, 9])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_near_matched_capacity_and_unique_comparison_anchor(self):
        configs = build_plan()['configurations']
        matched = [c['parameters'] for c in configs if 'matched_parameters' in c['groups']]
        self.assertLess(max(matched) / min(matched), 1.025)
        anchors = [c for c in configs if c['id'] == 'd2_w32_b128']
        self.assertEqual(len(anchors), 1)
        self.assertEqual(set(anchors[0]['groups']), {'depth', 'width', 'batch'})


if __name__ == '__main__':
    unittest.main()
