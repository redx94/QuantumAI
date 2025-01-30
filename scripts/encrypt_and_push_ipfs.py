#!/usr/bin/env python3

"""
scripts/encrypt_and_push_ipfs.py

Encrypts a specified file using AES-256 (CBC mode) and uploads the
encrypted file to IPFS. Prints the IPFS hash and the AES key/IV for decryption.

Dependencies:
  pip install pycryptodome py-ipfs-http-client

Usage Example:
  python scripts/encrypt_and_push_ipfs.py \
      --file "path/to/model_weights.bin" \
      --out "model_weights.enc"
"""

import os
import argparse
import ipfshttpclient
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

def encrypt_file_aes_cbc(input_file: str, output_file: str):
    """
    Encrypts 'input_file' using AES-256 in CBC mode and writes
    the ciphertext to 'output_file'. Returns (key, iv).
    """
    # Read plaintext from file
    with open(input_file, "rb") as f:
        plaintext = f.read()

    # Generate random 256-bit key & 128-bit IV
    key = get_random_bytes(32)  # AES-256
    iv = get_random_bytes(16)

    # Create AES cipher in CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

    # Pad the plaintext to a multiple of the block size
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

    # Write IV + ciphertext to the output file
    with open(output_file, "wb") as f:
        f.write(iv + ciphertext)

    return key, iv

def push_to_ipfs(file_path: str) -> str:
    """
    Pushes the specified file to IPFS and returns the IPFS hash (CID).
    Note: Requires an IPFS daemon running locally or a remote IPFS node.
    """
    client = ipfshttpclient.connect()  # Connect to IPFS daemon (127.0.0.1:5001 by default)
    res = client.add(file_path)
    return res["Hash"]

def main():
    parser = argparse.ArgumentParser(description="Encrypt file and upload to IPFS")
    parser.add_argument("--file", required=True, help="Path to input file to encrypt")
    parser.add_argument("--out", required=True, help="Path to output encrypted file")
    args = parser.parse_args()

    input_file = args.file
    output_file = args.out

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # 1) Encrypt local file with AES-256 (CBC)
    key, iv = encrypt_file_aes_cbc(input_file, output_file)

    # 2) Push encrypted file to IPFS
    ipfs_hash = push_to_ipfs(output_file)

    print("======================================================")
    print("Encryption & IPFS Upload Complete!")
    print(f" Original file:    {input_file}")
    print(f" Encrypted file:   {output_file}")
    print(f" IPFS Hash (CID):  {ipfs_hash}")
    print("------------------------------------------------------")
    print("To decrypt locally, you need the AES key and IV below.")
    print(f" AES Key (hex): {key.hex()}")
    print(f" IV (hex):      {iv.hex()}")
    print("======================================================")

if __name__ == "__main__":
    main()
