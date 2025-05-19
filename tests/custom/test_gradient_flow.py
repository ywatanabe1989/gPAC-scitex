#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test to verify gradient flow in DifferentiableBandPassFilter

import torch
import pytest
from gpac._DifferenciableBandPassFilter import DifferentiableBandPassFilter

def test_gradients_flow_through_filter():
    """Test that gradients properly flow through the DifferentiableBandPassFilter."""
    # Setup parameters
    seq_len = 1000
    fs = 1000.0
    pha_low_hz = 4.0
    pha_high_hz = 16.0
    pha_n_bands = 3
    amp_low_hz = 80.0
    amp_high_hz = 160.0
    amp_n_bands = 2
    
    # Create differentiable filter
    filter_module = DifferentiableBandPassFilter(
        sig_len=seq_len,
        fs=fs,
        pha_low_hz=pha_low_hz,
        pha_high_hz=pha_high_hz,
        pha_n_bands=pha_n_bands,
        amp_low_hz=amp_low_hz,
        amp_high_hz=amp_high_hz,
        amp_n_bands=amp_n_bands,
        cycle=3,
        fp16=False
    )
    
    # Verify parameters are set up correctly
    assert isinstance(filter_module.pha_mids, torch.nn.Parameter)
    assert isinstance(filter_module.amp_mids, torch.nn.Parameter)
    assert filter_module.pha_mids.requires_grad
    assert filter_module.amp_mids.requires_grad
    
    # Create input signal (batch=2, channels=1, time=seq_len)
    x = torch.randn(2, 1, seq_len)
    
    # Forward pass
    y = filter_module(x)
    
    # Create a loss that depends on all outputs
    loss = y.abs().mean()
    
    # Backward pass
    loss.backward()
    
    # Verify gradients
    assert filter_module.pha_mids.grad is not None
    assert filter_module.amp_mids.grad is not None
    
    # Check that gradients are non-zero
    assert not torch.all(filter_module.pha_mids.grad == 0)
    assert not torch.all(filter_module.amp_mids.grad == 0)
    
    # Print gradient values for debugging
    print("PHA gradients:", filter_module.pha_mids.grad)
    print("AMP gradients:", filter_module.amp_mids.grad)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])