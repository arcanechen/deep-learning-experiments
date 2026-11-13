"""无需下载数据的科学计算检查：划分、预处理、模型维度与梯度。"""
import unittest
import numpy as np
import torch
from torch import nn

from data import preprocess, stratified_indices
from models import MODEL_NAMES, build_model


class CoreTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_amp_training_updates_weights_and_validation_does_not(self):
        from train import run_epoch, seed_everything
        from torch.utils.data import DataLoader, TensorDataset
        seed_everything(42)
        device = torch.device('cuda')
        model = build_model('lenet').to(device)
        loader = DataLoader(TensorDataset(torch.randint(0, 256, (16, 1, 28, 28), dtype=torch.uint8),
                                          torch.arange(16) % 10), batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler('cuda')
        before = next(model.parameters()).detach().clone()
        metrics = run_epoch(model, loader, device, optimizer, scaler, amp=True)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertFalse(torch.equal(before, next(model.parameters())))
        trained = next(model.parameters()).detach().clone()
        run_epoch(model, loader, device, amp=True)
        torch.testing.assert_close(trained, next(model.parameters()), rtol=0, atol=0)
        torch.use_deterministic_algorithms(False)

    def test_alexnet_pool_matches_adaptive_for_fixed_input(self):
        x = torch.randn(2, 3, 7, 7, requires_grad=True)
        reference = nn.AdaptiveAvgPool2d((6, 6))(x)
        actual = nn.AvgPool2d(2, 1)(x)
        torch.testing.assert_close(actual, reference)
        gradient = torch.randn_like(actual)
        expected_grad = torch.autograd.grad(reference, x, gradient, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, x, gradient)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_models_support_strict_deterministic_backward(self):
        from train import seed_everything
        seed_everything(42)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name).cuda()
                loss = nn.CrossEntropyLoss()(model(torch.randn(2, 1, 32, 32, device='cuda')),
                                             torch.tensor([0, 9], device='cuda'))
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                del model, loss
        torch.use_deterministic_algorithms(False)

    def test_split_is_disjoint_complete_stratified_and_repeatable(self):
        labels = np.repeat(np.arange(10), 100)
        train, val = stratified_indices(labels)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(sorted(np.concatenate([train, val]).tolist()), list(range(1000)))
        np.testing.assert_array_equal(np.bincount(labels[val]), np.repeat(10, 10))
        np.testing.assert_array_equal(val, stratified_indices(labels)[1])

    def test_preprocessing_range_and_shape(self):
        x = torch.stack([torch.zeros(1, 28, 28, dtype=torch.uint8),
                         torch.full((1, 28, 28), 255, dtype=torch.uint8)])
        result = preprocess(x, 'cpu')
        self.assertEqual(tuple(result.shape), (2, 1, 32, 32))
        torch.testing.assert_close(result[0], torch.full((1, 32, 32), -1.0))
        torch.testing.assert_close(result[1], torch.ones(1, 32, 32))

    def test_all_models_have_valid_logits_and_gradients(self):
        torch.set_num_threads(4)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name)
                x = torch.randn(2, 1, 32, 32)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 10))
                loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 9]))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                print(f'{name}: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
                del model, logits, loss, gradients


if __name__ == '__main__':
    unittest.main()
