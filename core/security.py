from cryptography.fernet import Fernet
from ..config.config import config
import jwt
from datetime import datetime, timedelta

class SecurityManager:
    def __init__(self):
        self.key = config.SECRET_KEY.encode()
        self._fernet = Fernet(Fernet.generate_key())

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self._fernet.decrypt(encrypted_data.encode()).decode()

    def generate_token(self, user_id: str, expiry_hours: int = 24) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expiry_hours)
        }
        return jwt.encode(payload, self.key, algorithm='HS256')

    def verify_token(self, token: str) -> dict:
        """Verify JWT token"""
        return jwt.decode(token, self.key, algorithms=['HS256'])
