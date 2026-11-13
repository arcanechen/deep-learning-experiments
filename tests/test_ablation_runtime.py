import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ablation_common import atomic_checkpoint, rng_state, restore_rng, epoch, exclusive_lock
from controlled_cnn import ControlledCNN
from train import seed_everything


class RecoveryTests(unittest.TestCase):
    def test_cpu_resume_matches_uninterrupted_adam_dropout_shuffle(self):
        torch.set_num_threads(2)
        seed_everything(42)
        data = TensorDataset(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32, torch.arange(8) % 2)
        def setup():
            m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.2), nn.Linear(8, 2))
            return m, torch.optim.Adam(m.parameters()), torch.Generator().manual_seed(42)
        def one_epoch(m, o, g):
            for x, y in DataLoader(data, batch_size=2, shuffle=True, generator=g):
                o.zero_grad()
                nn.functional.cross_entropy(m(x), y).backward()
                o.step()
        m, o, g = setup()
        one_epoch(m, o, g)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            atomic_checkpoint(path, {'model': m.state_dict(), 'optimizer': o.state_dict(), 'rng': rng_state(g)})
            one_epoch(m, o, g)
            expected_random = (random.random(), np.random.rand(), torch.rand(3))
            restored = torch.load(path, map_location='cpu', weights_only=True)
            m2, o2, g2 = setup()
            m2.load_state_dict(restored['model'])
            o2.load_state_dict(restored['optimizer'])
            restore_rng(restored['rng'], g2)
            one_epoch(m2, o2, g2)
            for a, b in zip(m.parameters(), m2.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)

    def test_lock_prevents_duplicate_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'
            with exclusive_lock(path):
                with self.assertRaises(OSError):
                    with exclusive_lock(path):
                        pass
            with exclusive_lock(path):
                pass

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_updates_and_scaler_checkpoint(self):
        seed_everything(42)
        device = torch.device('cuda')
        model = ControlledCNN(1, 8).to(device)
        loader = DataLoader(TensorDataset(torch.randint(0, 256, (8, 1, 28, 28), dtype=torch.uint8),
                            torch.arange(8)), batch_size=4)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler('cuda')
        metrics = epoch(model, loader, device, optimizer, scaler)
        self.assertEqual(metrics['attempted_updates'], 2)
        self.assertEqual(metrics['actual_updates'], 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        epoch(model, loader, device)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        copy = torch.amp.GradScaler('cuda')
        copy.load_state_dict(scaler.state_dict())
        self.assertEqual(copy.state_dict(), scaler.state_dict())


if __name__ == '__main__':
    unittest.main()
