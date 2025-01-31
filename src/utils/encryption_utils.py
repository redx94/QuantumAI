
import os
import json
import base64
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from web3 import Web3

# Decrypt the data using AES
def decrypt_data(encrypted_data, key):
    cipher = AES.new(key, AES.MODE_CBC, iv=encrypted_data[:16])
    decrypted_data = unpad(cipher.decrypt(encrypted_data[16:]), AES.block_size)
    return decrypted_data

# Encrypt the data using AES
def encrypt_data(data, key):
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_data = cipher.iv + cipher.encrypt(pad(data, AES.block_size))
    return encrypted_data

# Push the encrypted data to IPFS
def push_to_ipfs(data):
    response = requests.post('https://ipfs.infura.io:5001/api/v0/add', files={'file': data})
    ipfs_hash = response.json()['Hash']
    return ipfs_hash

# Main function to decrypt data
def main_decrypt():
    encrypted_file_path = 'path/to/encrypted/file'
    key = b'your-encryption-key'

    with open(encrypted_file_path, 'rb') as f:
        encrypted_data = f.read()

    decrypted_data = decrypt_data(encrypted_data, key)
    print("Decrypted data:", decrypted_data.decode())

# Main function to encrypt and push data to IPFS
def main_encrypt_and_push():
    data = b'your-data-to-encrypt'
    key = b'your-encryption-key'

    encrypted_data = encrypt_data(data, key)
    ipfs_hash = push_to_ipfs(encrypted_data)
    print("IPFS hash:", ipfs_hash)

if __name__ == "__main__":
    # Uncomment the function you want to run
    # main_decrypt()
    # main_encrypt_and_push()