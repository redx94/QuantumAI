# QuantumAI

[Edit in StackBlitz next generation editor ⚡️](https://stackblitz.com/~/github.com/redx94/QuantumAI)

## Encrypting and Uploading Files to IPFS

The repository includes a script to securely encrypt files using AES-256 and upload them to IPFS:

1. **Install required libraries**:
   ```bash
   pip install pycryptodome py-ipfs-http-client
   ```

2. **Run the script**:
   ```bash
   python scripts/encrypt_and_push_ipfs.py --file path/to/original_model.bin --out path/to/encrypted_output.enc
   ```

The script will output:
- The IPFS hash (CID) of the encrypted file
- The AES key and IV needed for decryption

To decrypt files, you'll need both the IPFS CID and the encryption keys provided by the script.

### Decrypting Files from IPFS

To decrypt files downloaded from IPFS, use the `decrypt.py` script:

```bash
python scripts/decrypt.py \
    --encrypted downloaded.enc \
    --output decrypted.bin \
    --key <AES_KEY_FROM_ENCRYPTION> \
    --iv <IV_FROM_ENCRYPTION>
```

The script requires the original AES key and IV that were generated during encryption.

---

## Step-by-Step Explanation and Execution Guide

### 1. Installing IPFS on macOS

1. **Install IPFS using Homebrew**:
   ```bash
   brew install ipfs
   ```

2. **Initialize the IPFS Repository**:
   ```bash
   ipfs init
   ```
   - Creates a local IPFS configuration in `~/.ipfs`.

3. **Start the IPFS Daemon**:
   ```bash
   ipfs daemon
   ```
   - Keep this terminal open. The daemon runs on `localhost:5001`.

### 2. Verifying IPFS Functionality

1. **Check Node Information**:
   ```bash
   ipfs id
   ```
   - Displays your peer ID and addresses.

2. **Add a Test File**:
   ```bash
   echo "Hello IPFS" > test.txt
   ipfs add test.txt
   ```
   - Outputs the Content Identifier (CID), e.g., `QmXarR6rgkQ2fDSHjSY5nM2kuCXKYKRikyujLehRkLqSQU`.

3. **Retrieve the File**:
   ```bash
   ipfs cat QmXarR6rgkQ2fDSHjSY5nM2kuCXKYKRikyujLehRkLqSQU
   ```
   - Output: `Hello IPFS`.

### 3. Setting Up the Encryption Script

1. **Install Required Libraries**:
   ```bash
   pip install pycryptodome py-ipfs-http-client
   ```

2. **Create the Script `encrypt_and_push_ipfs.py`**:
   ```python
   from Crypto.Cipher import AES
   from Crypto.Util.Padding import pad
   from Crypto.Random import get_random_bytes
   import argparse
   import ipfshttpclient
   import os

   def encrypt_file(file_path, output_path):
       key = get_random_bytes(32)  # AES-256
       iv = get_random_bytes(16)   # Initialization Vector
       cipher = AES.new(key, AES.MODE_CBC, iv=iv)
       
       with open(file_path, 'rb') as f:
           plaintext = f.read()
       
       ciphertext = iv + cipher.encrypt(pad(plaintext, AES.block_size))
       
       with open(output_path, 'wb') as f:
           f.write(ciphertext)
       
       return key.hex(), iv.hex()

   def push_to_ipfs(file_path):
       client = ipfshttpclient.connect()
       res = client.add(file_path)
       return res['Hash']

   if __name__ == "__main__":
       parser = argparse.ArgumentParser(description='Encrypt a file and push to IPFS.')
       parser.add_argument('--file', required=True, help='Path to the file to encrypt.')
       parser.add_argument('--out', required=True, help='Output path for encrypted file.')
       args = parser.parse_args()

       key_hex, iv_hex = encrypt_file(args.file, args.out)
       cid = push_to_ipfs(args.out)
       
       print(f"File encrypted and uploaded to IPFS with CID: {cid}")
       print(f"AES Key (hex): {key_hex}")
       print(f"IV (hex): {iv_hex}")
   ```

### 4. Encrypting and Uploading a File

1. **Create a Dummy Model File**:
   ```bash
   dd if=/dev/urandom of=dummy_model.bin bs=1M count=10
   ```

2. **Run the Script**:
   ```bash
   python encrypt_and_push_ipfs.py --file dummy_model.bin --out dummy.enc
   ```

   **Output**:
   ```
   File encrypted and uploaded to IPFS with CID: QmXYZ...
   AES Key (hex): 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
   IV (hex): 0123456789abcdef0123456789abcdef
   ```

3. **Verify IPFS Upload**:
   ```bash
   ipfs cat QmXYZ... > downloaded.enc
   ```

### 5. Decrypting the File

1. **Save the AES Key and IV** from the script output.

2. **Decrypt Using This Script** (`decrypt.py`):
   ```python
   from Crypto.Cipher import AES
   from Crypto.Util.Padding import unpad
   import argparse

   def decrypt_file(encrypted_path, output_path, key_hex, iv_hex):
       key = bytes.fromhex(key_hex)
       iv = bytes.fromhex(iv_hex)
       
       with open(encrypted_path, 'rb') as f:
           ciphertext = f.read()
       
       cipher = AES.new(key, AES.MODE_CBC, iv=iv)
       decrypted = unpad(cipher.decrypt(ciphertext[16:]), AES.block_size)
       
       with open(output_path, 'wb') as f:
           f.write(decrypted)

   if __name__ == "__main__":
       parser = argparse.ArgumentParser(description='Decrypt a file.')
       parser.add_argument('--encrypted', required=True, help='Path to encrypted file.')
       parser.add_argument('--output', required=True, help='Output path for decrypted file.')
       parser.add_argument('--key', required=True, help='AES key in hex.')
       parser.add_argument('--iv', required=True, help='IV in hex.')
       args = parser.parse_args()

       decrypt_file(args.encrypted, args.output, args.key, args.iv)
   ```

3. **Run Decryption**:
   ```bash
   python decrypt.py --encrypted dummy.enc --output decrypted.bin --key 0123456789abcdef... --iv 0123456789abcdef...
   ```

4. **Verify Integrity**:
   ```bash
   shasum -a 256 dummy_model.bin decrypted.bin
   ```
   - Both hashes should match.

### 6. Pinning for Persistence

1. **Sign Up for [Pinata](https://pinata.cloud/)** and get an API key.

2. **Pin Your CID**:
   ```bash
   curl -X POST "https://api.pinata.cloud/pinning/pinByHash" \
        -H "Authorization: Bearer YOUR_JWT" \
        -H "Content-Type: application/json" \
        -d '{"hashToPin": "QmXYZ...", "pinataMetadata": {"name": "dummy-model"}}'
   ```

### 7. Security Best Practices

- **Store Keys Securely**: Use a password manager or encrypted vault.
- **Environment Variables**: Store keys in `.env` files (add `.env` to `.gitignore`).
- **Post-Quantum Encryption**: Wrap AES keys with Kyber (NIST PQC standard).

**Final Note**: You’ve successfully created a decentralized, encrypted storage system for AI models. Integrate this into your Quantum AI pipeline for secure, unstoppable distribution! 🚀

## Complete Installation and Usage Guide

### Installing IPFS (by Platform)

**macOS**:
```bash
brew install ipfs
ipfs init
ipfs daemon  # Run in a separate terminal
```

**Linux**:
```bash
# Using package manager
sudo apt install ipfs  # Ubuntu/Debian
sudo pacman -S ipfs    # Arch Linux

# Or download from IPFS
wget https://dist.ipfs.io/go-ipfs/v0.12.0/go-ipfs_v0.12.0_linux-amd64.tar.gz
tar -xvzf go-ipfs_v0.12.0_linux-amd64.tar.gz
cd go-ipfs
sudo bash install.sh
ipfs init
ipfs daemon  # Run in a separate terminal
```

**Windows**:
1. Download from https://dist.ipfs.io/#go-ipfs
2. Extract and run `ipfs.exe init`
3. Run `ipfs.exe daemon`

### Testing Your IPFS Setup

```bash
# Check node status
ipfs id

# Test with a sample file
echo "Hello IPFS" > test.txt
ipfs add test.txt
ipfs cat <received-hash>
```

### Using the Encryption Scripts

1. **Install Python Dependencies**:
   ```bash
   pip install pycryptodome py-ipfs-http-client
   ```

2. **Encrypt and Upload**:
   ```bash
   python scripts/encrypt_and_push_ipfs.py \
       --file your_model.bin \
       --out encrypted.bin
   ```
   Save the output AES key and IV!

3. **Download and Decrypt**:
   ```bash
   python scripts/decrypt.py \
       --encrypted downloaded.enc \
       --output decrypted.bin \
       --key <AES_KEY> \
       --iv <IV>
   ```

### File Persistence with Pinning

Register at [Pinata](https://pinata.cloud/) and use their API to pin your files:

```bash
export PINATA_JWT="your-jwt-token"
curl -X POST "https://api.pinata.cloud/pinning/pinByHash" \
     -H "Authorization: Bearer $PINATA_JWT" \
     -H "Content-Type: application/json" \
     -d '{"hashToPin": "<your-ipfs-hash>"}'
```

### Security Best Practices

1. **Key Management**:
   - Store encryption keys in a password manager
   - Use `.env` files for development (add to `.gitignore`)
   - Consider hardware security modules (HSM) for production

2. **Post-Quantum Security**:
   - The scripts use AES-256 which is quantum-resistant
   - Consider wrapping keys with Kyber for future-proofing

3. **Access Control**:
   - Implement key rotation policies
   - Use separate keys for different model versions
   - Consider multi-party computation for key sharing