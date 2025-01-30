terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

module "quantum_cluster" {
  source = "./modules/quantum-cluster"
  
  cluster_name = "quantum-ai-prod"
  region      = var.aws_region
  node_groups = {
    quantum_workers = {
      desired_size = 3
      max_size     = 10
      min_size     = 1
      instance_types = ["p4d.24xlarge"]  // GPU instances for quantum simulation
    }
  }
}

module "quantum_api" {
  source = "./modules/api-gateway"
  
  name = "quantum-ai-api"
  stage = "prod"
  throttling_rate_limit = 10000
  throttling_burst_limit = 5000
}
