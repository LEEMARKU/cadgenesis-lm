"""tests/multimodal/test_encoders.py
====================================
Unit tests for the vision and point cloud modality encoders.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.pointcloud import PointCloudEncoder
from cadgenesis.multimodal.encoders.vision import VisionEncoderCNN


class TestVisionEncoderCNN:
    def test_modality_contract(self):
        enc = VisionEncoderCNN()
        assert enc.modality is Modality.IMAGE
        assert enc.sequence_aware is False
        assert enc.feature_dim == 512

    def test_forward_square_input(self):
        enc = VisionEncoderCNN()
        out = enc.forward(torch.randn(2, 3, 224, 224))
        assert tuple(out.shape) == (2, 512)

    def test_forward_non_square_input(self):
        enc = VisionEncoderCNN()
        out = enc.forward(torch.randn(2, 3, 112, 320))
        assert tuple(out.shape) == (2, 512)

    def test_forward_wrong_dim_raises(self):
        enc = VisionEncoderCNN()
        with pytest.raises(ValueError):
            enc.forward(torch.randn(2, 3, 224))

    def test_encode_tensor_batch(self):
        enc = VisionEncoderCNN()
        out = enc.encode(torch.randn(2, 3, 32, 32))
        assert tuple(out.shape) == (2, 512)

    def test_encode_single_tensor(self):
        enc = VisionEncoderCNN()
        out = enc.encode(torch.randn(3, 32, 32))
        assert tuple(out.shape) == (1, 512)

    def test_encode_list_of_tensors(self):
        enc = VisionEncoderCNN()
        images = [torch.randn(3, 32, 32) for _ in range(2)]
        out = enc.encode(images)
        assert tuple(out.shape) == (2, 512)

    def test_encode_nested_list(self):
        enc = VisionEncoderCNN()
        image = [[[0.25] * 16 for _ in range(16)] for _ in range(3)]
        out = enc.encode(image)
        assert tuple(out.shape) == (1, 512)

    def test_backward_flows_to_conv(self):
        enc = VisionEncoderCNN()
        enc.forward(torch.randn(2, 3, 64, 64)).sum().backward()
        assert enc.net[0].weight.grad is not None

    def test_parameters_and_finite(self):
        enc = VisionEncoderCNN()
        assert sum(p.numel() for p in enc.parameters()) > 0
        out = enc.forward(torch.randn(2, 3, 32, 32))
        assert torch.isfinite(out).all()


class TestPointCloudEncoder:
    def test_modality_contract(self):
        enc = PointCloudEncoder()
        assert enc.modality is Modality.POINT_CLOUD
        assert enc.sequence_aware is False
        assert enc.feature_dim == 512

    def test_forward(self):
        enc = PointCloudEncoder()
        out = enc.forward(torch.randn(2, 1024, 3))
        assert tuple(out.shape) == (2, 512)

    def test_forward_variable_points(self):
        enc = PointCloudEncoder()
        out = enc.forward(torch.randn(2, 500, 3))
        assert tuple(out.shape) == (2, 512)

    def test_forward_wrong_dim_raises(self):
        enc = PointCloudEncoder()
        with pytest.raises(ValueError):
            enc.forward(torch.randn(2, 3))
        with pytest.raises(ValueError):
            enc.forward(torch.randn(2, 1024, 3, 1))

    def test_forward_wrong_point_dim_raises(self):
        enc = PointCloudEncoder(point_dim=3)
        with pytest.raises(ValueError):
            enc.forward(torch.randn(2, 1024, 4))

    def test_encode_tensor_batch(self):
        enc = PointCloudEncoder()
        out = enc.encode(torch.randn(2, 1024, 3))
        assert tuple(out.shape) == (2, 512)

    def test_encode_list_of_tensors(self):
        enc = PointCloudEncoder()
        clouds = [torch.randn(1024, 3) for _ in range(2)]
        out = enc.encode(clouds)
        assert tuple(out.shape) == (2, 512)

    def test_backward_flows_to_first_linear(self):
        enc = PointCloudEncoder()
        enc.forward(torch.randn(2, 512, 3)).sum().backward()
        assert enc.mlp[0].weight.grad is not None

    def test_parameters_and_finite(self):
        enc = PointCloudEncoder()
        assert sum(p.numel() for p in enc.parameters()) > 0
        out = enc.forward(torch.randn(2, 256, 3))
        assert torch.isfinite(out).all()
