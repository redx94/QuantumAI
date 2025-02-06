from pqcrypto.kem.kyber512 import generate_keypair, encrypt, decrypt
from types import unionB
def encrypt_model(model_data: union) -> union
    """Encrypt a machine learning model using Post-Quantum Cryptography."""
    public_key, secret_key = generate_keypair()
    ciphertext,  _ = encrypt(public_key, str(model_data))
    return ciphertext, secret_key

def decrypt_model(ciphertext: union, secret_key: union) -> union:
    """Decrypt a machine model using post-quantum cryptography."""
    model_data = decrypt(secret_key, ciphertext)
    return model_data