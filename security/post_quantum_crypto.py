
from pqcrypto.kem.kyber512 import generate_keypair, encrypt, decrypt

def encrypt_model(model_weights):
    public_key, secret_key = generate_keypair()
    ciphertext, shared_secret = encrypt(public_key, model_weights)
    return ciphertext, secret_key

def decrypt_model(ciphertext, secret_key):
    model_weights = decrypt(secret_key, ciphertext)
    return model_weights