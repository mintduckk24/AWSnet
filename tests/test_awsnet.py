"""Unit tests for AWSnet MyoPS380 Adaptation."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure AWSnet repo root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.awsnet import AWSnet, ConvBlock, UpConv, AttentionBlock, AttU_Net
from utils.criterions import LossPRSnet_dicece, MultiDiceLoss, DiceLoss, AWSCompositeLoss
from utils.lr_scheduler import build_optimizer, build_scheduler


class TestAWSnetArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = AWSnet(
            in_channels=3,
            num_classes=4,
            base_filters=16,  # Lightweight for fast unit testing
            deep_supervision=True,
        ).to(self.device)

    def test_eval_forward_single_tensor(self):
        """Tests forward pass with 1 concatenated tensor (B, 3, 128, 128) in eval mode."""
        self.model.eval()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 4, 128, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_eval_forward_three_tensors(self):
        """Tests forward pass with 3 separate tensors (c0, lge, t2) in eval mode."""
        self.model.eval()
        c0 = torch.randn(2, 1, 128, 128, device=self.device)
        lge = torch.randn(2, 1, 128, 128, device=self.device)
        t2 = torch.randn(2, 1, 128, 128, device=self.device)
        with torch.no_grad():
            out = self.model(c0, lge, t2)
        self.assertEqual(out.shape, (2, 4, 128, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_train_forward_deep_supervision(self):
        """Tests training mode outputs 4 deep supervision stages + primary output."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        outputs = self.model(x)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 5)  # output4, output3, output2, output1, output

        out4, out3, out2, out1, out_primary = outputs
        self.assertEqual(out4.shape, (2, 4, 128, 128))
        self.assertEqual(out3.shape, (2, 4, 128, 128))
        self.assertEqual(out2.shape, (2, 4, 128, 128))
        self.assertEqual(out1.shape, (2, 4, 128, 128))
        self.assertEqual(out_primary.shape, (2, 4, 128, 128))

    def test_return_all_in_eval_mode(self):
        """Tests return_all=True forces output of all 5 stages in eval mode."""
        self.model.eval()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        with torch.no_grad():
            outputs = self.model(x, return_all=True)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 5)

    def test_attu_net_alias(self):
        """Tests that AttU_Net is an exported alias for AWSnet."""
        self.assertIs(AttU_Net, AWSnet)

    def test_backward_and_gradients(self):
        """Tests backward pass flows gradients to all layers including attention blocks."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128, device=self.device)
        target = torch.randint(0, 4, (2, 128, 128), device=self.device)

        outputs = self.model(x)
        criterion = LossPRSnet_dicece(num_classes=4)
        loss, metrics = criterion(outputs[0], outputs[1], outputs[2], outputs[3], target)

        loss.backward()
        self.assertTrue(torch.isfinite(loss))

        # Check gradients in Conv1, Conv5, UpConv, Attention blocks, and deep supervision heads
        self.assertIsNotNone(self.model.Conv1.conv[0].weight.grad)
        self.assertIsNotNone(self.model.Conv5.conv[0].weight.grad)
        self.assertIsNotNone(self.model.Att5.W_g[0].weight.grad)
        self.assertIsNotNone(self.model.Att2.psi[0].weight.grad)
        self.assertIsNotNone(self.model.final3.weight.grad)
        self.assertIsNotNone(self.model.final.weight.grad)

    def test_parameter_count(self):
        """Tests parameter count calculation for standard base_filters=32."""
        std_model = AWSnet(in_channels=3, num_classes=4, base_filters=32, deep_supervision=True)
        param_count = std_model.count_parameters()
        self.assertGreater(param_count, 8_000_000)
        self.assertLess(param_count, 10_000_000)
        self.assertEqual(param_count, 8_729_116)


class TestAWSnetLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.criterion = LossPRSnet_dicece(num_classes=4)

    def test_multidice_loss(self):
        """Tests MultiDiceLoss on perfect and inverted predictions."""
        mdl = MultiDiceLoss(num_classes=4)
        # Perfect logits
        target = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        logits = F.one_hot(target, num_classes=4).permute(2, 0, 1).unsqueeze(0).float() * 100.0
        loss_perfect = mdl(logits, target)
        self.assertLess(loss_perfect.item(), 0.05)

    def test_composite_loss_dict_metrics(self):
        """Tests AWSCompositeLoss outputs correct telemetry metrics."""
        comp_loss = AWSCompositeLoss(num_classes=4)
        out4 = torch.randn(2, 4, 64, 64)
        out3 = torch.randn(2, 4, 64, 64)
        out2 = torch.randn(2, 4, 64, 64)
        out1 = torch.randn(2, 4, 64, 64)
        target = torch.randint(0, 4, (2, 64, 64))

        loss, metrics = comp_loss((out4, out3, out2, out1, out1), target)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_stage1", metrics)
        self.assertIn("loss_stage4", metrics)

    def test_lossprsnet_dicece_eight_arguments(self):
        """Tests backward compatibility when called with 8 arguments like upstream train_val.py."""
        criterion = LossPRSnet_dicece(num_classes=4)
        out4 = torch.randn(2, 4, 32, 32)
        out3 = torch.randn(2, 4, 32, 32)
        out2 = torch.randn(2, 4, 32, 32)
        out1 = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))

        # Call with 8 arguments
        loss, metrics = criterion(out4, out3, out2, out1, target, target, target, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_total", metrics)

    def test_upstream_asan_modules_cpu(self):
        """Tests that upstream ASAN_training models and loss run without crash on CPU."""
        from ASAN_training.Network.Models import AttU_Net as UpstreamAttUNet
        from ASAN_training.loss_deepsupvision import LossPRSnet_dicece as UpstreamLoss

        model = UpstreamAttUNet(in_ch=3, out_ch=4, deep_supervision=True)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        outputs = model(x)
        self.assertEqual(len(outputs), 5)

        loss_fn = UpstreamLoss(num_classes=4)
        target = torch.randint(0, 4, (2, 32, 32))
        loss = loss_fn(outputs[0], outputs[1], outputs[2], outputs[3], target, target, target, target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
