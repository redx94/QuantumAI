import subprocess

# Deploy API using Cloudflare Workers
def deploy_cloudflare():
    print("Deploying to Cloudflare Workers...")
    subprocess.run(["wrangler", "publish"])

# Deploy API using Hugging Face Spaces
def deploy_huggingface():
    print("Deploying to Hugging Face Spaces...")
    subprocess.run(["huggingface-cli", "repo", "create", "QuantumAI"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Deploy Quantum API"])
    subprocess.run(["git", "push", "origin", "main"])

if __name__ == "__main__":
    deploy_cloudflare()
    deploy_huggingface()
