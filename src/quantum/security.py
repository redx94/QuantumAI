import os
from cryptography.fernet import Fernet

class SecurityManager:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def encrypt(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())

    def decrypt(self, token: bytes) -> str:
        return self.cipher.decrypt(token).decode()

class SecureKeyManager:
    def __init__(self):
        self._load_keys()
        
    def _load_keys(self):
        self.api_key = os.getenv('QUANTUM_API_KEY')
        if not self.api_key:
            raise ValueError("QUANTUM_API_KEY environment variable not set")
