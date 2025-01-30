#!/usr/bin/env python3
"""
Decrypts files that were encrypted using encrypt_and_push_ipfs.py.
Requires the AES key and IV from the original encryption.

Dependencies:
  pip install pycryptodome

Example Usage:
  python scripts/decrypt.py \
      --encrypted downloaded.enc \
      --output decrypted.bin \
      --key <hex_key> \
      --iv <hex_iv>
"""

import argparse
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def decrypt_file(encrypted_path: str, output_path: str, key_hex: str, iv_hex: str):
    """
    Decrypt a file that was encrypted with AES-256 CBC mode.
    """
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    
    with open(encrypted_path, 'rb') as f:
        ciphertext = f.read()
    
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted = unpad(cipher.decrypt(ciphertext[16:]), AES.block_size)
    
    with open(output_path, 'wb') as f:
        f.write(decrypted)

def main():
    parser = argparse.ArgumentParser(description='Decrypt a file.')
    parser.add_argument('--encrypted', required=True, help='Path to encrypted file')
    parser.add_argument('--output', required=True, help='Output path for decrypted file')
    parser.add_argument('--key', required=True, help='AES key in hex format')
    parser.add_argument('--iv', required=True, help='IV in hex format')
    args = parser.parse_args()

    decrypt_file(args.encrypted, args.output, args.key, args.iv)
    print(f"File successfully decrypted to: {args.output}")

if __name__ == "__main__":
    main()
