#!/usr/bin/env python3
"""
Test script for Linear Attention modules.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import LinearAttention, PerformerAttention, LinearAttentionBlock

def test_linear_attention():
    """Test LinearAttention module."""
    print("Testing LinearAttention...")

    # Create test input
    x = torch.randn(2, 64, 8, 8)  # batch=2, channels=64, height=8, width=8

    # Create module
    attn = LinearAttention(dim=64, num_heads=8, attn_ratio=0.5)

    # Forward pass
    output = attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print("✓ LinearAttention test passed\n")

def test_performer_attention():
    """Test PerformerAttention module."""
    print("Testing PerformerAttention...")

    # Create test input
    x = torch.randn(2, 64, 8, 8)

    # Create module
    attn = PerformerAttention(dim=64, num_heads=8, attn_ratio=0.5, nb_features=32)

    # Forward pass
    output = attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print("✓ PerformerAttention test passed\n")

def test_linear_attention_block():
    """Test LinearAttentionBlock module."""
    print("Testing LinearAttentionBlock...")

    # Create test input
    x = torch.randn(2, 64, 8, 8)

    # Create module
    block = LinearAttentionBlock(dim=64, num_heads=8, attn_ratio=0.5, mlp_ratio=2.0)

    # Forward pass
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print("✓ LinearAttentionBlock test passed\n")

def test_gradient_flow():
    """Test that gradients flow properly through the modules."""
    print("Testing gradient flow...")

    x = torch.randn(2, 64, 8, 8, requires_grad=True)

    # Test LinearAttention
    attn = LinearAttention(dim=64, num_heads=8)
    output = attn(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients not computed for LinearAttention"
    print("✓ LinearAttention gradient flow test passed")

    # Reset gradients
    x.grad.zero_()

    # Test PerformerAttention
    attn = PerformerAttention(dim=64, num_heads=8)
    output = attn(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients not computed for PerformerAttention"
    print("✓ PerformerAttention gradient flow test passed")

    # Reset gradients
    x.grad.zero_()

    # Test LinearAttentionBlock
    block = LinearAttentionBlock(dim=64, num_heads=8)
    output = block(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients not computed for LinearAttentionBlock"
    print("✓ LinearAttentionBlock gradient flow test passed\n")

if __name__ == "__main__":
    print("Running Linear Attention Module Tests\n")

    try:
        test_linear_attention()
        test_performer_attention()
        test_linear_attention_block()
        test_gradient_flow()

        print("🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise