import subprocess
from web3 import Web3
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import docker
from typing import Optional, Dict
import yaml
import requests
import logging
from docker.models.containers import Container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QuantumAPIDeployer:
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.api_image_name = "quantum-ai-api"
        self.api_container_name = "quantum-ai-api-container"

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def build_docker_image(self, dockerfile_path: str = "Dockerfile") -> None:
        """Build Docker image for the Quantum AI API"""
        try:
            logger.info("Building Docker image...")
            self.docker_client.images.build(
                path=str(Path(dockerfile_path).parent),
                dockerfile=dockerfile_path,
                tag=self.api_image_name
            )
            logger.info("Docker image built successfully")
        except Exception as e:
            logger.error(f"Error building Docker image: {e}")
            raise

    def deploy_container(
        self,
        port: int = 8000,
        env_vars: Optional[Dict[str, str]] = None
    ) -> Container:
        """Deploy the API container"""
        try:
            # Stop existing container if running
            self.stop_container()
            
            logger.info("Starting API container...")
            container = self.docker_client.containers.run(
                self.api_image_name,
                name=self.api_container_name,
                environment=env_vars or {},
                ports={'8000/tcp': port},
                detach=True
            )
            
            logger.info(f"API container started successfully. ID: {container.id}")
            return container
            
        except Exception as e:
            logger.error(f"Error deploying container: {e}")
            raise

    def stop_container(self) -> None:
        """Stop and remove existing container if running"""
        try:
            container = self.docker_client.containers.get(self.api_container_name)
            container.stop()
            container.remove()
            logger.info("Existing container stopped and removed")
        except docker.errors.NotFound:
            pass

    def deploy_to_cloud(
        self,
        cloud_provider: str,
        credentials: Dict[str, str]
    ) -> Dict[str, str]:
        """Deploy to cloud provider (AWS, GCP, or Azure)"""
        supported_providers = ['aws', 'gcp', 'azure']
        if cloud_provider.lower() not in supported_providers:
            raise ValueError(f"Unsupported cloud provider. Use: {supported_providers}")
            
        # Implementation would vary based on cloud provider
        # This is a placeholder for the cloud deployment logic
        logger.info(f"Deploying to {cloud_provider}...")
        
        return {
            "status": "deployed",
            "provider": cloud_provider,
            "endpoint": f"https://api.{cloud_provider}.example.com/quantum"
        }

    def health_check(self, endpoint: str) -> bool:
        """Verify API health after deployment"""
        try:
            response = requests.get(f"{endpoint}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def setup_monitoring(self):
        print("Setting up monitoring...")
        # Add monitoring setup (e.g., Prometheus + Grafana)
        pass

    def setup_ci_cd(self):
        print("Setting up CI/CD pipeline...")
        # Create GitHub Actions workflow
        workflow_path = ".github/workflows/deploy.yml"
        os.makedirs(os.path.dirname(workflow_path), exist_ok=True)
        
        workflow_content = """
name: Deploy Quantum AI API

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
        """
        with open(workflow_path, 'w') as f:
            f.write(workflow_content)

if __name__ == "__main__":
    deployer = QuantumAPIDeployer()
    
    # Build and deploy locally
    deployer.build_docker_image("Dockerfile")
    container = deployer.deploy_container(
        port=8000,
        env_vars={"API_KEY": os.getenv("API_KEY")}
    )
    
    # Verify deployment
    if deployer.health_check("http://localhost:8000"):
        logger.info("Deployment successful!")
    else:
        logger.error("Deployment verification failed!")

    deployer.deploy_to_cloud("cloudflare")
    deployer.deploy_to_cloud("huggingface")
    deployer.setup_monitoring()
    deployer.setup_ci_cd()
