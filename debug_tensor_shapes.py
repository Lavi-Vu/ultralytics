#!/usr/bin/env python3
import torch
import torch.nn as nn
from ultralytics.nn.modules import LinearAttention, PerformerAttention

def debug_shapes():
    print("Debugging tensor shapes...")

    # Create test input
    x = torch.randn(2, 64, 8, 8)  # batch=2, channels=64, height=8, width=8
    print(f"Input shape: {x.shape}")

    B, C, H, W = x.shape
    N = H * W
    print(f"B={B}, C={C}, H={H}, W={W}, N={N}")

    # Test LinearAttention
    print("\n--- LinearAttention ---")
    attn = LinearAttention(dim=64, num_heads=8, attn_ratio=0.5)
    print(f"num_heads: {attn.num_heads}")
    print(f"head_dim: {attn.head_dim}")
    print(f"key_dim: {attn.key_dim}")
    print(f"value_dim: {attn.value_dim}")

    # Project to query, key, value
    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    print(f"q shape after proj: {q.shape}")
    print(f"k shape after proj: {k.shape}")
    print(f"v shape after proj: {v.shape}")

    q = q.view(B, attn.num_heads, attn.key_dim, N)
    k = k.view(B, attn.num_heads, attn.key_dim, N)
    v = v.view(B, attn.num_heads, attn.value_dim, N)
    print(f"q shape after view: {q.shape}")
    print(f"k shape after view: {k.shape}")
    print(f"v shape after view: {v.shape}")

    # Apply feature maps
    q = attn._feature_map(q)
    k = attn._feature_map(k)
    print(f"q shape after feature map: {q.shape}")
    print(f"k shape after feature map: {k.shape}")

    # Compute KV^T
    kv = torch.einsum('bhkn,bhvm->bhkvm', k, v)
    print(f"kv shape: {kv.shape}")

    # Compute Q(KV)
    out = torch.einsum('bhkn,bhkvm->bhvm', q, kv)
    print(f"out shape after Q(KV): {out.shape}")

    # Compute normalization
    try:
        z = 1 / (torch.einsum('bhkn,bhkn->bhn', q, k) + attn.eps)
        print(f"z shape: {z.shape}")
        out = out * z.unsqueeze(1)
        print(f"out shape after normalization: {out.shape}")
    except Exception as e:
        print(f"Error in normalization: {e}")

    # Test PerformerAttention
    print("\n--- PerformerAttention ---")
    performer = PerformerAttention(dim=64, num_heads=8, attn_ratio=0.5, nb_features=32)
    print(f"num_heads: {performer.num_heads}")
    print(f"head_dim: {performer.head_dim}")
    print(f"key_dim: {performer.key_dim}")
    print(f"value_dim: {performer.value_dim}")
    print(f"nb_features: {performer.nb_features}")

    # Project to query, key, value
    q = performer.q_proj(x)
    k = performer.k_proj(x)
    v = performer.v_proj(x)
    print(f"q shape after proj: {q.shape}")
    print(f"k shape after proj: {k.shape}")
    print(f"v shape after proj: {v.shape}")

    q = q.view(B, performer.num_heads, performer.key_dim, N)
    k = k.view(B, performer.num_heads, performer.key_dim, N)
    v = v.view(B, performer.num_heads, performer.value_dim, N)
    print(f"q shape after view: {q.shape}")
    print(f"k shape after view: {k.shape}")
    print(f"v shape after view: {v.shape}")

    # Apply random feature maps
    print(f"w_q shape: {performer.w_q.shape}")
    print(f"w_k shape: {performer.w_k.shape}")

    try:
        q_prime = torch.einsum('bhjn,hin->bhi', q, performer.w_q)
        print(f"q_prime shape: {q_prime.shape}")
    except Exception as e:
        print(f"Error in q_prime einsum: {e}")
        print(f"q shape: {q.shape}")
        print(f"w_q shape: {performer.w_q.shape}")
        # Try different einsum notation
        q_prime = torch.einsum('bhik,hkj->bhij', q, performer.w_q)
        print(f"q_prime shape with bhik,hkj->bhij: {q_prime.shape}")

if __name__ == "__main__":
    debug_shapes()