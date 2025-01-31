
from security.post_quantum_crypto import encrypt_model, decrypt_model
import numpy as np

model_weights = np.random.random(100)
ciphertext, secret_key = encrypt_model(model_weights)
decrypted_weights = decrypt_model(ciphertext, secret_key)

assert np.allclose(model_weights, decrypted_weights)