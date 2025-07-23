# MinIO Installation & Configuration Guide

## Overview

This guide provides complete instructions for setting up MinIO on a Debian 12 VM with external SSD storage, Tailscale networking, and Let's Encrypt HTTPS certificates for GitLab LFS integration.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Host Machine  │    │   MinIO VM      │    │  GitLab Server  │
│                 │    │  (Debian 12)    │    │                 │
│ /media/shared   │◄──►│ /mnt/external   │◄──►│   LFS Client    │
│ drive (SSD)     │    │ -ssd            │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌───────▼───────┐
                         │   Tailscale   │
                         │   Network     │
                         │ (HTTPS/TLS)   │
                         └───────────────┘
```

## Prerequisites

- Host machine with KVM/libvirt virtualization
- External SSD mounted at `/media/shared drive` on host
- Tailscale account and authentication key
- GitLab instance requiring LFS storage

## Part 1: VM Creation and Base Setup

### 1.1 Create Debian 12 VM

```bash
# Create VM with libvirt/KVM
virt-install \
  --name minio-storage \
  --ram 4096 \
  --disk path=/var/lib/libvirt/images/minio-storage.qcow2,size=20 \
  --vcpus 2 \
  --os-variant debian12 \
  --network bridge=virbr0 \
  --graphics none \
  --console pty,target_type=serial \
  --location 'https://deb.debian.org/debian/dists/bookworm/main/installer-amd64/' \
  --extra-args 'console=ttyS0,115200n8 serial'
```

### 1.2 Configure Shared Storage Access

**On Host Machine:**

Edit VM configuration to add virtiofs filesystem sharing:

```bash
virsh edit minio-storage
```

Add this section to the VM XML:

```xml
<filesystem type='mount' accessmode='passthrough'>
  <driver type='virtiofs'/>
  <source dir='/media/shared drive'/>
  <target dir='external-ssd'/>
</filesystem>
```

Restart the VM:

```bash
virsh shutdown minio-storage
virsh start minio-storage
```

## Part 2: Debian 12 Base Configuration

### 2.1 System Update and Essential Packages

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget gnupg lsb-release ca-certificates \
                    htop ncdu tree vim fuse3 jq

# Configure sudo for user (if needed)
sudo usermod -aG sudo $USER
```

### 2.2 Configure External SSD Mount

```bash
# Create mount point
sudo mkdir -p /mnt/external-ssd

# Mount shared folder
sudo mount -t virtiofs external-ssd /mnt/external-ssd

# Verify mount
df -h | grep external-ssd
ls -la /mnt/external-ssd

# Configure auto-mount
echo 'external-ssd /mnt/external-ssd virtiofs defaults 0 0' | sudo tee -a /etc/fstab

# Test auto-mount
sudo umount /mnt/external-ssd
sudo mount -a
df -h | grep external-ssd
```

### 2.3 Prepare MinIO Directories

```bash
# Create MinIO directories
sudo mkdir -p /mnt/external-ssd/minio-data
sudo mkdir -p /mnt/external-ssd/minio-config
sudo mkdir -p /mnt/external-ssd/minio-certs

# Create minio user with fixed UID/GID
sudo groupadd -g 1001 minio
sudo useradd -u 1001 -g 1001 -s /bin/false -d /mnt/external-ssd/minio-data minio

# Set permissions
sudo chown -R minio:minio /mnt/external-ssd/minio-data
sudo chown -R minio:minio /mnt/external-ssd/minio-config
sudo chown -R minio:minio /mnt/external-ssd/minio-certs
sudo chmod 755 /mnt/external-ssd/minio-data
```

## Part 3: Docker Installation

### 3.1 Install Docker CE on Debian 12

```bash
# Remove old versions
sudo apt remove -y docker docker-engine docker.io containerd runc

# Add Docker repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
                    docker-buildx-plugin docker-compose-plugin

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER

# Verify installation
sudo docker --version
sudo docker run hello-world

# Note: Logout and login again to apply docker group
```

## Part 4: Tailscale Configuration

### 4.1 Install Tailscale

```bash
# Add Tailscale repository
curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.list | \
  sudo tee /etc/apt/sources.list.d/tailscale.list

# Install Tailscale
sudo apt update
sudo apt install -y tailscale

# Start Tailscale with required features
sudo tailscale up --accept-routes --accept-dns --advertise-tags=tag:minio-server

# Note: Follow authentication URL provided by the command above
```

### 4.2 Configure HTTPS Certificates

```bash
# Get Tailscale hostname
TAILSCALE_HOSTNAME=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')
echo "Tailscale hostname: $TAILSCALE_HOSTNAME"

# Request Let's Encrypt certificate via Tailscale
sudo tailscale cert --domain $TAILSCALE_HOSTNAME

# Copy certificates for MinIO
sudo cp /var/lib/tailscale/certs/${TAILSCALE_HOSTNAME}.crt /mnt/external-ssd/minio-certs/public.crt
sudo cp /var/lib/tailscale/certs/${TAILSCALE_HOSTNAME}.key /mnt/external-ssd/minio-certs/private.key
sudo chown minio:minio /mnt/external-ssd/minio-certs/*
sudo chmod 600 /mnt/external-ssd/minio-certs/private.key
sudo chmod 644 /mnt/external-ssd/minio-certs/public.crt
```

### 4.3 Certificate Auto-Renewal Script

```bash
# Create certificate update script
cat > ~/update-minio-certs.sh << 'EOF'
#!/bin/bash
TAILSCALE_HOSTNAME=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')

# Renew certificate
sudo tailscale cert --domain $TAILSCALE_HOSTNAME

# Copy new certificates
sudo cp /var/lib/tailscale/certs/${TAILSCALE_HOSTNAME}.crt /mnt/external-ssd/minio-certs/public.crt
sudo cp /var/lib/tailscale/certs/${TAILSCALE_HOSTNAME}.key /mnt/external-ssd/minio-certs/private.key
sudo chown minio:minio /mnt/external-ssd/minio-certs/*
sudo chmod 600 /mnt/external-ssd/minio-certs/private.key
sudo chmod 644 /mnt/external-ssd/minio-certs/public.crt

# Restart MinIO to apply new certificates
cd ~/minio-setup
docker compose restart minio

echo "Certificates updated for $TAILSCALE_HOSTNAME"
EOF

chmod +x ~/update-minio-certs.sh

# Setup weekly certificate renewal
echo "0 2 * * 0 $HOME/update-minio-certs.sh >> /var/log/minio-cert-update.log 2>&1" | crontab -
```

## Part 5: MinIO Deployment

### 5.1 Docker Compose Configuration

```bash
# Create working directory
mkdir -p ~/minio-setup
cd ~/minio-setup

# Get Tailscale hostname for configuration
TAILSCALE_HOSTNAME=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')

# Create docker-compose.yml
cat > docker-compose.yml << EOF
version: '3.8'

services:
  minio:
    image: quay.io/minio/minio:RELEASE.2024-01-16T16-07-38Z
    container_name: minio-lfs
    restart: always
    user: "1001:1001"
    ports:
      - "9000:9000"   # API HTTPS
      - "9001:9001"   # Console HTTPS
    volumes:
      - /mnt/external-ssd/minio-data:/data
      - /mnt/external-ssd/minio-config:/root/.minio
      - /mnt/external-ssd/minio-certs:/root/.minio/certs
    environment:
      MINIO_ROOT_USER: gitlab-lfs-admin
      MINIO_ROOT_PASSWORD: SuperSecurePassword123!
      MINIO_BROWSER_REDIRECT_URL: https://${TAILSCALE_HOSTNAME}:9001
      MINIO_SERVER_URL: https://${TAILSCALE_HOSTNAME}:9000
      MINIO_DOMAIN: ${TAILSCALE_HOSTNAME}
    command: server /data --console-address ":9001" --certs-dir /root/.minio/certs
    healthcheck:
      test: ["CMD", "curl", "-f", "-k", "https://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
      start_period: 120s
    networks:
      - minio-network

networks:
  minio-network:
    driver: bridge
EOF
```

### 5.2 Deploy MinIO

```bash
# Start MinIO
docker compose up -d

# Verify deployment
docker compose ps
docker compose logs minio

# Wait for service to be ready
sleep 60
docker compose logs minio | tail -20
```

## Part 6: MinIO Configuration

### 6.1 Install and Configure MinIO Client

```bash
# Install MinIO Client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Get Tailscale hostname
TAILSCALE_HOSTNAME=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')

# Configure MinIO client alias
mc alias set minio-secure https://${TAILSCALE_HOSTNAME}:9000 gitlab-lfs-admin SuperSecurePassword123!

# Test connection
mc admin info minio-secure
```

### 6.2 Setup GitLab LFS Integration

```bash
# Create GitLab LFS user
mc admin user add minio-secure gitlab-lfs-user GitLabLFS2025!

# Create LFS policy
cat > lfs-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetObjectVersion",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::mfc-project-lfs",
        "arn:aws:s3:::mfc-project-lfs/*"
      ]
    }
  ]
}
EOF

# Apply policy
mc admin policy create minio-secure lfs-policy lfs-policy.json
mc admin policy attach minio-secure lfs-policy --user gitlab-lfs-user

# Create LFS bucket with versioning
mc mb minio-secure/mfc-project-lfs
mc version enable minio-secure/mfc-project-lfs

# Test upload
echo "Test LFS upload $(date)" | mc pipe minio-secure/mfc-project-lfs/test-lfs.txt
mc ls minio-secure/mfc-project-lfs/
```

## Part 7: Security Configuration

### 7.1 Firewall Setup

```bash
# Install and configure UFW
sudo apt install -y ufw

# Configure default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential services
sudo ufw allow ssh

# Allow Tailscale traffic
sudo ufw allow in on tailscale0

# Allow MinIO ports only from Tailscale
sudo ufw allow in on tailscale0 to any port 9000 comment 'MinIO API via Tailscale'
sudo ufw allow in on tailscale0 to any port 9001 comment 'MinIO Console via Tailscale'

# Enable firewall
sudo ufw --force enable

# Verify configuration
sudo ufw status verbose
```

## Part 8: Monitoring and Maintenance

### 8.1 Health Check Script

```bash
# Create monitoring script
cat > ~/minio-health-check.sh << 'EOF'
#!/bin/bash
TAILSCALE_HOSTNAME=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')

echo "=== MinIO Health Check $(date) ==="
echo "Tailscale Status:"
tailscale status | grep -E "(Self|minio)"

echo -e "\nMinIO HTTPS Health:"
curl -s -k https://${TAILSCALE_HOSTNAME}:9000/minio/health/live

echo -e "\nMinIO Container Status:"
cd ~/minio-setup
docker compose ps

echo -e "\nCertificate Expiry:"
echo | openssl s_client -servername ${TAILSCALE_HOSTNAME} -connect ${TAILSCALE_HOSTNAME}:9000 2>/dev/null | openssl x509 -noout -dates

echo -e "\nStorage Usage:"
du -sh /mnt/external-ssd/minio-data/

echo -e "\nBucket Status:"
mc ls minio-secure/ 2>/dev/null || echo "MinIO not accessible"
EOF

chmod +x ~/minio-health-check.sh

# Setup hourly monitoring
echo "0 * * * * $HOME/minio-health-check.sh >> /var/log/minio-health.log 2>&1" | crontab -

# Test health check
~/minio-health-check.sh
```

### 8.2 Backup Script

```bash
# Create backup script for configuration
cat > ~/backup-minio-config.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/mnt/external-ssd/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup MinIO configuration
tar -czf $BACKUP_DIR/minio-config-$DATE.tar.gz \
  -C /mnt/external-ssd minio-config \
  -C ~/minio-setup docker-compose.yml \
  -C ~ update-minio-certs.sh minio-health-check.sh

# Keep only last 7 backups
ls -t $BACKUP_DIR/minio-config-*.tar.gz | tail -n +8 | xargs rm -f

echo "Configuration backup completed: minio-config-$DATE.tar.gz"
EOF

chmod +x ~/backup-minio-config.sh

# Setup daily backup
echo "0 3 * * * $HOME/backup-minio-config.sh >> /var/log/minio-backup.log 2>&1" | crontab -
```

## Part 9: GitLab Integration

### 9.1 GitLab Configuration

Add the following to `/etc/gitlab/gitlab.rb` on your GitLab server:

```ruby
# MinIO LFS Configuration with Tailscale HTTPS
gitlab_rails['lfs_enabled'] = true
gitlab_rails['lfs_object_store_enabled'] = true
gitlab_rails['lfs_object_store_remote_directory'] = "mfc-project-lfs"
gitlab_rails['lfs_object_store_connection'] = {
  'provider' => 'AWS',
  'region' => 'us-east-1',
  'aws_access_key_id' => 'gitlab-lfs-user',
  'aws_secret_access_key' => 'GitLabLFS2025!',
  'endpoint' => 'https://YOUR_TAILSCALE_HOSTNAME:9000',
  'path_style' => true,
  'aws_signature_version' => 4,
  'use_ssl' => true,
  'ssl_verify_peer' => true
}

# Performance optimizations
gitlab_rails['lfs_object_store_background_upload'] = true
gitlab_rails['lfs_object_store_proxy_download'] = false
gitlab_rails['lfs_object_store_direct_upload'] = true
```

### 9.2 Apply GitLab Configuration

```bash
# On GitLab server
sudo gitlab-ctl reconfigure
sudo gitlab-ctl restart
sudo gitlab-ctl status
```

## Part 10: Testing and Validation

### 10.1 MinIO Functionality Test

```bash
# Test file upload
echo "Test file content $(date)" | mc pipe minio-secure/mfc-project-lfs/test-upload.txt

# Test file download
mc cat minio-secure/mfc-project-lfs/test-upload.txt

# Test bucket listing
mc ls minio-secure/mfc-project-lfs/

# Test versioning
echo "Updated content $(date)" | mc pipe minio-secure/mfc-project-lfs/test-upload.txt
mc ls --versions minio-secure/mfc-project-lfs/test-upload.txt
```

### 10.2 HTTPS Certificate Validation

```bash
# Check certificate details
TAILSCALE_HOSTNAME=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')
echo | openssl s_client -servername ${TAILSCALE_HOSTNAME} -connect ${TAILSCALE_HOSTNAME}:9000 2>/dev/null | openssl x509 -noout -text

# Test HTTPS connectivity
curl -I https://${TAILSCALE_HOSTNAME}:9000/minio/health/live
```

## Access Information

After successful installation:

- **MinIO Console**: `https://[tailscale-hostname]:9001`
- **MinIO API**: `https://[tailscale-hostname]:9000`
- **Admin User**: `gitlab-lfs-admin`
- **Admin Password**: `SuperSecurePassword123!`
- **GitLab LFS User**: `gitlab-lfs-user`
- **GitLab LFS Password**: `GitLabLFS2025!`

## Troubleshooting

### Common Issues

1. **Certificate Issues**:
   ```bash
   # Regenerate certificates
   sudo tailscale cert --domain $(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')
   ~/update-minio-certs.sh
   ```

2. **Storage Permission Issues**:
   ```bash
   # Fix ownership
   sudo chown -R minio:minio /mnt/external-ssd/minio-data
   ```

3. **Docker Issues**:
   ```bash
   # Restart MinIO
   cd ~/minio-setup
   docker compose down
   docker compose up -d
   ```

4. **Tailscale Connectivity**:
   ```bash
   # Restart Tailscale
   sudo systemctl restart tailscaled
   sudo tailscale up --accept-routes --accept-dns
   ```

## Maintenance Tasks

### Weekly Tasks
- Certificate renewal (automated via cron)
- Health check review
- Storage usage monitoring

### Monthly Tasks
- System updates
- Log rotation
- Backup verification

### Quarterly Tasks
- Security audit
- Performance optimization
- Disaster recovery testing

## Security Considerations

- All traffic encrypted via Tailscale HTTPS
- Access restricted to Tailscale network
- Regular certificate renewal
- Firewall protection
- Storage on external SSD for easy backup/recovery
- Separate user accounts for different access levels

## Performance Optimization

- SSD storage for high I/O performance
- Docker resource limits if needed
- MinIO tuning for large files
- Network optimization for Tailscale

---

**Version**: 1.0  
**Last Updated**: 2025-01-23  
**Tested On**: Debian 12 (Bookworm)  
**MinIO Version**: RELEASE.2024-01-16T16-07-38Z