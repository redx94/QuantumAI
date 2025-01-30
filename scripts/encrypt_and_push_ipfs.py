#!/usr/bin/env python3
"""
scripts/encrypt_and_push_ipfs.py

Encrypts a specified file using AES-256 in CBC mode and uploads the
encrypted file to IPFS. Prints the IPFS hash and the AES key/IV for decryption.

Dependencies:
  pip install pycryptodome py-ipfs-http-client

Example Usage:
  python scripts/encrypt_and_push_ipfs.py \
      --file path/to/model_weights.bin \
      --out path/to/model_weights.enc
"""

import os
import argparse
import ipfshttpclient
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

def encrypt_file_aes(filepath, output_path):
    key = get_random_bytes(32)
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

    with open(filepath, "rb") as f:
        plaintext = f.read()

    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

    with open(output_path, "wb") as fo:
        fo.write(iv + ciphertext)

    return key, iv

def push_to_ipfs(file_path):
    client = ipfshttpclient.connect()
    res = client.add(file_path)
    return res["Hash"]

def main():
    model_file = "quantum_mistral_ft/pytorch_model.bin"
    out_file = "quantum_mistral_ft/weights.enc"
    
    key, iv = encrypt_file_aes(model_file, out_file)
    ipfs_hash = push_to_ipfs(out_file)
    
    print(f"IPFS hash: {ipfs_hash}")
    print(f"AES key: {key.hex()}")
    print(f"IV: {iv.hex()}")

if __name__ == "__main__":
    main()
