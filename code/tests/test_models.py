"""Tests for src.models — written FIRST (TDD). Verifies forward shapes and
parameter counts (LSTM must equal the verified 6,177; others within sane
ranges of Paper 1's search space)."""

import torch

from src.models import AeDetector, LstmDetector, TcnDetector, TransformerDetector, make_model


def test_lstm_shape_and_exact_params():
    m = LstmDetector()
    out = m(torch.zeros(8, 3, 14))
    assert out.shape == (8,)
    assert sum(p.numel() for p in m.parameters()) == 6177  # == Keras 6,049 + PyTorch double-bias 128


def test_tcn_shape_and_param_range():
    m = TcnDetector()
    assert m(torch.zeros(8, 3, 14)).shape == (8,)
    n = sum(p.numel() for p in m.parameters())
    assert 3_000 <= n <= 15_000


def test_transformer_shape_and_param_range():
    m = TransformerDetector(window=3)
    assert m(torch.zeros(8, 3, 14)).shape == (8,)
    n = sum(p.numel() for p in m.parameters())
    assert 5_000 <= n <= 30_000


def test_autoencoder_reconstructs_and_scores():
    m = AeDetector()
    x = torch.randn(8, 3, 14)
    rec = m(x)
    assert rec.shape == (8, 3, 14)
    s = m.score(x)
    assert s.shape == (8,) and (s >= 0).all()
    n = sum(p.numel() for p in m.parameters())
    assert 1_500 <= n <= 8_000


def test_make_model_factory():
    for name, cls in [("lstm", LstmDetector), ("tcn", TcnDetector),
                      ("transformer", TransformerDetector), ("ae", AeDetector)]:
        assert isinstance(make_model(name, window=3), cls)
