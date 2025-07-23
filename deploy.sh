#!/bin/bash
set -e

# MinIO Deployment Script
echo "=== MinIO Ansible Deployment ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check requirements
echo -e "${YELLOW}Checking requirements...${NC}"

if ! command -v ansible &> /dev/null; then
    echo -e "${RED}Ansible not found. Install with: pip install ansible${NC}"
    exit 1
fi

if ! command -v ansible-vault &> /dev/null; then
    echo -e "${RED}Ansible Vault not found.${NC}"
    exit 1
fi

# Install required collections
echo -e "${YELLOW}Installing Ansible collections...${NC}"
ansible-galaxy collection install -r requirements.yml

# Check inventory
if [ ! -f inventory.yml ]; then
    echo -e "${RED}inventory.yml not found. Copy from inventory.yml.example and configure.${NC}"
    exit 1
fi

# Check vault file
if [ ! -f vault.yml ]; then
    echo -e "${YELLOW}vault.yml not found. Creating from example...${NC}"
    if [ -f vault.yml.example ]; then
        cp vault.yml.example vault.yml
        echo -e "${YELLOW}Please edit vault.yml with your credentials and encrypt it:${NC}"
        echo "ansible-vault encrypt vault.yml"
        exit 1
    else
        echo -e "${RED}vault.yml.example not found${NC}"
        exit 1
    fi
fi

# Test connectivity
echo -e "${YELLOW}Testing connectivity to target hosts...${NC}"
if ansible minio_servers -m ping --ask-vault-pass; then
    echo -e "${GREEN}Connectivity test passed${NC}"
else
    echo -e "${RED}Connectivity test failed${NC}"
    exit 1
fi

# Confirm deployment
read -p "Ready to deploy MinIO? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Run deployment
echo -e "${YELLOW}Starting MinIO deployment...${NC}"
ansible-playbook ansible-minio-setup.yml --ask-vault-pass

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo
    echo "Next steps:"
    echo "1. Access MinIO Console at: https://[tailscale-hostname]:9001"
    echo "2. Configure GitLab LFS with the MinIO endpoint"
    echo "3. Run health check: ansible minio_servers -m shell -a '~/minio-health-check.sh'"
else
    echo -e "${RED}Deployment failed. Check logs above.${NC}"
    exit 1
fi