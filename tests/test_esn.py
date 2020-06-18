import matplotlib.pyplot as plt
import jax.numpy as jnp

from esn.esn import sparse_esncell, apply_sparse_esn, generate_state_matrix

def test_sparse_esn():
    esn = sparse_esncell(1,100)
    inputs = jnp.ones((100,1))
    X  = generate_state_matrix(esn, inputs, 2)
    assert X.shape == (98, 102)

if __name__ == "__main__":
    test_sparse_esn()