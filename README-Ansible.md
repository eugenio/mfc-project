# MinIO Ansible Deployment

Automated deployment of MinIO LFS server with Tailscale HTTPS on Debian 12 VM.

## Prerequisites

1. **Ansible Control Node**:
   ```bash
   pip install ansible
   ansible-galaxy install -r requirements.yml
   ```

2. **Target VM**:
   - Debian 12 with SSH access
   - External SSD mounted via virtiofs as `external-ssd`
   - User with sudo privileges

3. **Tailscale Auth Key**:
   - Generate at: https://login.tailscale.com/admin/settings/keys
   - One-time use recommended

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repo-url>
   cd minio-ansible
   ```

2. **Configure Inventory**:
   ```bash
   # Edit inventory.yml with your VM details
   vim inventory.yml
   ```

3. **Setup Secrets**:
   ```bash
   # Create vault file
   ansible-vault create vault.yml
   # Add your secrets (see vault.yml.example)
   ```

4. **Deploy**:
   ```bash
   # Test connectivity
   ansible minio_servers -m ping
   
   # Deploy MinIO
   ansible-playbook ansible-minio-setup.yml --ask-vault-pass
   ```

## Configuration Files

- `ansible-minio-setup.yml` - Main playbook
- `inventory.yml` - Host configuration
- `group_vars/all.yml` - Default variables
- `vault.yml` - Encrypted secrets (create from vault.yml.example)
- `templates/` - Jinja2 templates for configuration files

## Vault Variables

Create `vault.yml` with:
```yaml
vault_minio_root_password: "your-secure-password"
vault_gitlab_lfs_password: "your-lfs-password"
vault_tailscale_auth_key: "tskey-auth-xxxxx"
```

## Manual Steps After Deployment

1. **Verify Deployment**:
   ```bash
   # SSH to VM and check services
   ssh debian@your-vm-ip
   sudo docker ps
   tailscale status
   mc ls minio-secure/
   ```

2. **Access MinIO Console**:
   - URL: `https://[tailscale-hostname]:9001`
   - User: `gitlab-lfs-admin`
   - Password: `[your-vault-password]`

3. **Configure GitLab**:
   - Update `/etc/gitlab/gitlab.rb` with MinIO endpoint
   - Run `sudo gitlab-ctl reconfigure`

## Customization

### Override Variables

In `inventory.yml` or `group_vars/`:
```yaml
minio_root_user: "custom-admin"
lfs_bucket_name: "custom-bucket"
minio_data_path: "/custom/path"
```

### Add Multiple VMs

In `inventory.yml`:
```yaml
minio_servers:
  hosts:
    minio-vm-1:
      ansible_host: 192.168.1.100
    minio-vm-2:
      ansible_host: 192.168.1.101
```

## Troubleshooting

### Connection Issues
```bash
# Test SSH connectivity
ansible minio_servers -m ping -u debian

# Check Tailscale status
ansible minio_servers -m shell -a "tailscale status"
```

### Service Issues
```bash
# Check Docker services
ansible minio_servers -m shell -a "docker compose -f ~/minio-setup/docker-compose.yml ps"

# Check MinIO logs
ansible minio_servers -m shell -a "docker compose -f ~/minio-setup/docker-compose.yml logs minio"
```

### Certificate Issues
```bash
# Regenerate certificates
ansible minio_servers -m shell -a "~/update-minio-certs.sh"
```

## Maintenance

### Run Health Checks
```bash
ansible minio_servers -m shell -a "~/minio-health-check.sh"
```

### Update MinIO
```bash
# Edit docker-compose.yml.j2 with new image version
# Re-run playbook
ansible-playbook ansible-minio-setup.yml --ask-vault-pass --tags=minio
```

### Backup Configuration
```bash
ansible minio_servers -m shell -a "~/backup-minio-config.sh"
```

## Security Notes

- All passwords stored in encrypted Ansible Vault
- HTTPS enforced via Tailscale certificates
- Firewall configured to allow only Tailscale traffic
- Regular certificate renewal automated
- Log rotation configured

## Support

For issues:
1. Check logs: `/var/log/minio-*.log`
2. Verify Tailscale connectivity
3. Check Docker container status
4. Review firewall rules with `sudo ufw status`