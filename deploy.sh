#!/bin/bash

# Load configuration
CONFIG_FILE="config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found!"
    exit 1
fi

# Extract Cloudflare credentials from config
CLOUDFLARE_API_TOKEN=$(jq -r '.cloudflare.api_token' "$CONFIG_FILE")
CLOUDFLARE_ZONE_ID=$(jq -r '.cloudflare.zone_id' "$CONFIG_FILE")
CLOUDFLARE_EMAIL=$(jq -r '.cloudflare.email' "$CONFIG_FILE")
WORKER_NAME=$(jq -r '.deployment.cloudflare_worker_name' "$CONFIG_FILE")

# Build the application
echo "Building application..."
npm run build

# Package the worker code
echo "Packaging worker code..."
wrangler pack ./dist

# Deploy to Cloudflare Workers
echo "Deploying to Cloudflare Workers..."
wrangler publish \
    --name "$WORKER_NAME" \
    --env production

# Purge Cloudflare cache
echo "Purging Cloudflare cache..."
curl -X POST "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/purge_cache" \
     -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
     -H "Content-Type: application/json" \
     --data '{"purge_everything":true}'

# Verify deployment
echo "Verifying deployment..."
WORKER_STATUS=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
    "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/workers/scripts/$WORKER_NAME")

if echo "$WORKER_STATUS" | grep -q "success":true; then
    echo "Deployment successful!"
else
    echo "Deployment verification failed!"
    exit 1
fi
