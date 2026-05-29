# Suno Discord Analysis — VPS Deployment Guide (CI/CD)

> **Convert to PDF**: In VS Code, open this file → right-click tab → *Open Preview* → `Ctrl+P` → *Save as PDF*.

---

## Overview

Every `git push` to `main` automatically:
1. Builds the React frontend
2. Builds the Rust backend as a Docker image → pushes to GitHub Container Registry
3. rsyncs the frontend files to your VPS
4. SSHes into the VPS → pulls the new image → restarts services

**Services running on VPS:**

```
nginx  (port 80 / 443) ─── serves frontend + proxies /api/* to backend
backend (internal only) ── Rust API on port 8000
postgres (internal only) ─ PostgreSQL database
qdrant (internal only) ─── vector store on port 6333
```

---

## Prerequisites

| Item | Details |
|---|---|
| VPS | Ubuntu 22.04 LTS, 2 GB RAM minimum, ports 22 / 80 / 443 open |
| Domain | A domain you control (e.g. `yourdomain.com`) |
| GitHub repo | This project pushed to GitHub |
| Local tools | `git`, `psql`, `scp` or `ssh` available on your machine |

---

## Step 1 — Point DNS to Your VPS

In your domain registrar or DNS control panel, create an **A record**:

```
Type: A
Name: @  (or yourdomain.com)
Value: <your VPS IP address>
TTL: 300
```

Wait for propagation before continuing Step 4 (SSL). Check with:

```bash
ping yourdomain.com
# Should reply from your VPS IP
```

---

## Step 2 — Install Docker on VPS

SSH into your VPS:

```bash
ssh user@your-vps-ip
```

Install Docker:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
docker --version    # verify
```

---

## Step 3 — Create a Deploy SSH Key (on your LOCAL machine)

This key lets GitHub Actions SSH into your VPS without a password.

```bash
# Run on your LOCAL machine
ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ""
```

This creates two files:
- `~/.ssh/deploy_key` → **private key** (goes into GitHub secret)
- `~/.ssh/deploy_key.pub` → **public key** (goes onto VPS)

Authorize the public key on the VPS:

```bash
# Still on your LOCAL machine — copy public key to VPS
ssh-copy-id -i ~/.ssh/deploy_key.pub user@your-vps-ip
```

Or manually:

```bash
# On VPS
echo "PASTE_CONTENT_OF_deploy_key.pub_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Test it works:

```bash
ssh -i ~/.ssh/deploy_key user@your-vps-ip "echo OK"
# Should print: OK
```

---

## Step 4 — Get SSL Certificate

> DNS must be pointing to your VPS before this step works.

On the **VPS** (Docker is not running yet, so port 80 is free):

```bash
sudo apt install -y certbot
sudo certbot certonly --standalone -d yourdomain.com
```

Certbot saves certificates to `/etc/letsencrypt/live/yourdomain.com/`. These are mounted read-only into the nginx container.

Verify auto-renewal works:

```bash
sudo certbot renew --dry-run
```

---

## Step 5 — Create App Directory and `.env` on VPS

On the **VPS**:

```bash
mkdir -p ~/app/frontend/dist
mkdir -p ~/app/nginx
```

Create the environment file:

```bash
nano ~/app/.env
```

Paste the following and fill in your values:

```env
# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_PASSWORD=choose_a_strong_password_here

# ── Qdrant (self-hosted container) ───────────────────────────────────────────
QDRANT_COLLECTION=discord_openai
QDRANT_API_KEY=

# ── Application ──────────────────────────────────────────────────────────────
APP_MODE=multi
OPENAI_API_KEY=sk-...

# ── Docker image (set automatically by CI, override here if needed) ──────────
BACKEND_IMAGE=ghcr.io/hafizhrafizal/retrieval-backend:latest

# ── Logging ──────────────────────────────────────────────────────────────────
RUST_LOG=info
```

Secure the file:

```bash
chmod 600 ~/app/.env
```

---

## Step 6 — Copy Config Files to VPS

On your **LOCAL machine** (from the project root):

```bash
# Copy docker-compose
scp docker-compose.yml user@your-vps-ip:~/app/docker-compose.yml

# Copy nginx config
scp nginx/nginx.conf user@your-vps-ip:~/app/nginx/nginx.conf
```

On the **VPS**, replace the domain placeholder in nginx config:

```bash
sed -i 's/YOUR_DOMAIN/yourdomain.com/g' ~/app/nginx/nginx.conf

# Verify the replacement
grep "server_name" ~/app/nginx/nginx.conf
# Should show: server_name yourdomain.com;
```

---

## Step 7 — Set GitHub Repository Secrets

In your GitHub repository, go to:
**Settings → Secrets and variables → Actions → New repository secret**

Add these four secrets:

| Secret name | Value |
|---|---|
| `VPS_HOST` | Your VPS IP address |
| `VPS_USER` | SSH username (e.g. `ubuntu` or `root`) |
| `VPS_SSH_KEY` | Full contents of `~/.ssh/deploy_key` (the private key — starts with `-----BEGIN OPENSSH PRIVATE KEY-----`) |
| `VPS_APP_DIR` | `/home/ubuntu/app` (the full path to your app directory on the VPS) |

To get the private key content:

```bash
cat ~/.ssh/deploy_key
# Copy everything including the BEGIN and END lines
```

---

## Step 8 — Push to Trigger First Deployment

On your **LOCAL machine**:

```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

Watch the build at:
`https://github.com/hafizhrafizal/Retrieval-Web-Refactor/actions`

The workflow runs three jobs in sequence:

| Job | What it does | Time |
|---|---|---|
| `build-frontend` | `npm install && npm run build`, uploads dist as artifact | ~2 min |
| `build-backend-image` | `docker build`, pushes to `ghcr.io` | ~10 min (first time), ~2 min (cached) |
| `deploy` | rsyncs frontend dist, SSHes into VPS, `docker pull` + `docker compose up -d` | ~1 min |

---

## Step 9 — Verify Services on VPS

SSH into the VPS and check:

```bash
cd ~/app
docker compose ps
```

Expected output (all services `Up`):

```
NAME               IMAGE                    STATUS
app-postgres-1     postgres:16-alpine       Up (healthy)
app-qdrant-1       qdrant/qdrant:latest     Up
app-backend-1      ...retrieval-backend     Up
app-nginx-1        nginx:alpine             Up
```

Check backend logs:

```bash
docker compose logs backend --tail=40
```

Look for these lines (confirms everything started correctly):

```
Database initialized
search_vector index is current — fast keyword search active from startup
Listening on http://0.0.0.0:8000
```

Test the API from the VPS:

```bash
curl http://localhost:8000/api/stats
# Should return JSON with total_messages, embedded_messages, etc.
```

Test the full stack through nginx:

```bash
curl https://yourdomain.com/api/stats
# Should return same JSON through HTTPS
```

Open in browser: `https://yourdomain.com`
→ The React app should load and show the login page.

---

## Step 10 — Migrate PostgreSQL (Local → VPS)

### 10.1 Dump the local database

On your **LOCAL machine**:

```bash
pg_dump -h localhost -U postgres -d discord_db \
  --no-owner --no-acl \
  -f discord_db_export.sql
```

> **Windows (PowerShell)**: if `pg_dump` is not found, use the full path:
> `& "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" -h localhost -U postgres -d discord_db --no-owner --no-acl -f discord_db_export.sql`

### 10.2 Transfer to VPS

```bash
scp discord_db_export.sql user@your-vps-ip:~/app/
```

### 10.3 Restore into the VPS database

On the **VPS**:

```bash
cd ~/app

# Copy dump file into the postgres container
docker compose cp discord_db_export.sql postgres:/tmp/discord_db_export.sql

# Restore (some notices about existing objects are normal — not errors)
docker compose exec postgres \
  psql -U retrieval -d retrieval -f /tmp/discord_db_export.sql

# Clean up
docker compose exec postgres rm /tmp/discord_db_export.sql
rm ~/app/discord_db_export.sql
```

### 10.4 Verify row counts match

```bash
# On VPS
docker compose exec postgres \
  psql -U retrieval -d retrieval \
  -c "SELECT COUNT(*) FROM messages;"
```

Compare against local:

```bash
# On LOCAL machine
psql -h localhost -U postgres -d discord_db \
  -c "SELECT COUNT(*) FROM messages;"
```

Both numbers must match.

---

## Step 11 — Migrate ChromaDB → Qdrant

The migration script reads your local ChromaDB files and pushes all vectors into the Qdrant container on the VPS via an SSH tunnel.

### 11.1 Install migration script dependencies (LOCAL machine)

```bash
pip install chromadb qdrant-client tqdm
```

### 11.2 Start Qdrant on VPS

On the **VPS**:

```bash
cd ~/app
docker compose up -d qdrant
docker compose ps qdrant    # should show Up
```

### 11.3 Open an SSH tunnel (LOCAL machine)

Qdrant runs inside Docker and is not exposed to the internet. Open a tunnel from your local port 6333 to the VPS Qdrant:

```bash
# Open this in a SEPARATE terminal — keep it running during the migration
ssh -L 6333:localhost:6333 user@your-vps-ip -N
```

Test the tunnel:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

### 11.4 Dry run first (count vectors, no upload)

```bash
cd /path/to/Retrieval-Web-Refactor

python scripts/migrate_chroma_to_qdrant.py \
  --chroma-path ./chroma_db \
  --chroma-collection discord_openai \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection discord_openai \
  --dry-run
```

Confirm the reported count matches your expected number of embedded messages.

### 11.5 Run the migration

```bash
python scripts/migrate_chroma_to_qdrant.py \
  --chroma-path ./chroma_db \
  --chroma-collection discord_openai \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection discord_openai \
  --vector-size 1536 \
  --batch-size 200
```

Expected output:

```
Connecting to ChromaDB at ./chroma_db ...
  Found 45,231 vectors in ChromaDB collection 'discord_openai'

Connecting to Qdrant at http://localhost:6333 ...
  Creating Qdrant collection 'discord_openai' (dim=1536, cosine) ...

Migrating 45,231 vectors in batches of 200 ...
100%|████████████████████████████| 227/227 [04:12<00:00]

Done! Uploaded 45,231 vectors. Errors: 0.
Qdrant reports 45,231 vectors in 'discord_openai'.
Migration verified successfully.
```

### 11.6 Close the SSH tunnel

Press `Ctrl+C` in the tunnel terminal.

---

## Step 12 — Final Verification

### Check the stats page

Open `https://yourdomain.com/api/stats` in your browser. Confirm:

```json
{
  "total_messages": 123456,
  "total_uploads": 5,
  "embedded_messages": 45231,
  "vector_db_label": "Qdrant",
  "api_key_set": true
}
```

- `total_messages` → matches your local PostgreSQL count
- `embedded_messages` → matches your ChromaDB vector count
- `vector_db_label` → must be `"Qdrant"` (confirms Qdrant is connected)

### Log in as admin

- URL: `https://yourdomain.com`
- Username: `admin`
- Password: `Admin@2025!`

### Test search

Run a keyword search and a semantic search to confirm the database and vectors are working.

---

## Ongoing Operations

### Deploy an update

Just push to `main` — everything is automated:

```bash
git add .
git commit -m "your change"
git push origin main
```

CI/CD takes ~3–5 min for frontend-only changes, ~3 min for backend changes (cached Rust build).

### View live logs

```bash
# SSH into VPS first
docker compose -f ~/app/docker-compose.yml logs -f backend
docker compose -f ~/app/docker-compose.yml logs -f nginx
```

### Restart a single service

```bash
docker compose -f ~/app/docker-compose.yml restart backend
```

### Backup PostgreSQL

```bash
cd ~/app
docker compose exec postgres \
  pg_dump -U retrieval retrieval > "backup_$(date +%Y%m%d_%H%M).sql"
```

### Renew SSL certificate

Certbot renews automatically. If you ever need to force-renew:

```bash
# On VPS — stop nginx to free port 80
cd ~/app && docker compose stop nginx
sudo certbot renew
docker compose start nginx
```

### Check disk space

```bash
docker system df                  # Docker images, volumes
du -sh ~/app/frontend/dist        # Frontend files
```

---

## Appendix — Reusing a Previously Used Domain

If your domain was already pointing somewhere (old server, old app, expired project), work through the scenarios below **before** running Step 1 and Step 4 of this guide.

---

### Scenario A — Domain was pointing to a different server

The old server is gone or you no longer need it. You just need to re-point DNS and get a fresh SSL cert.

**1. Update the DNS A record** to your new VPS IP (same as Step 1):

```
Type: A
Name: @
Value: <new VPS IP>
```

**2. Wait for propagation** — can take 1–48 hours depending on old TTL:

```bash
# Run repeatedly until it returns your new VPS IP
dig yourdomain.com +short
```

Or check online: https://dnschecker.org

**3. Continue from Step 2** of this guide. Certbot will issue a new certificate on the new VPS — the old server's certificate is irrelevant.

---

### Scenario B — Domain was used on this same VPS (different app)

Something else is already running on port 80 or 443 on the VPS.

**1. Check what is using port 80/443:**

```bash
sudo ss -tlnp | grep -E ':80|:443'
# or
sudo lsof -i :80
sudo lsof -i :443
```

**2. Stop the conflicting service:**

```bash
# If it's a systemd service (e.g. nginx, apache2):
sudo systemctl stop nginx
sudo systemctl stop apache2

# If it's another docker compose project:
cd /path/to/old-project && docker compose down

# If it's a plain Docker container:
docker ps                        # find the container name
docker stop <container-name>
```

**3. Continue from Step 4** (SSL certificate) of this guide.

---

### Scenario C — Certbot already has a certificate for this domain

If certbot was previously run on this VPS for the same domain, it keeps the old certificate in `/etc/letsencrypt`.

**Check existing certificates:**

```bash
sudo certbot certificates
```

You will see output like:

```
Found the following certs:
  Certificate Name: yourdomain.com
    Domains: yourdomain.com
    Expiry Date: 2025-09-01 (VALID: 45 days)
    Certificate Path: /etc/letsencrypt/live/yourdomain.com/fullchain.pem
```

**If the certificate is still valid** — skip Step 4 entirely. The existing cert is already at the correct path and nginx will use it automatically.

**If the certificate is expired** — renew it:

```bash
# Stop anything on port 80 first
sudo certbot renew --force-renewal -d yourdomain.com
```

**If you want to start completely fresh** — delete and reissue:

```bash
sudo certbot delete --cert-name yourdomain.com
sudo certbot certonly --standalone -d yourdomain.com
```

---

### Scenario D — Port 80 is blocked by a firewall or cloud panel

Some VPS providers (DigitalOcean, AWS, Hetzner) have a separate firewall in their web console that is independent of `ufw`.

**Check ufw (Ubuntu firewall):**

```bash
sudo ufw status
```

If port 80 or 443 is missing, add them:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload
```

**Check your VPS provider's firewall panel** — look for "Firewall", "Security Groups", or "Network" in the dashboard and ensure inbound rules allow TCP 80 and 443.

---

### Scenario E — Old nginx config conflicts with the new one

If a previous app left an nginx config mounted into a Docker container, the nginx container may fail to start because of port or `server_name` conflicts.

**Symptoms** — nginx container exits immediately:

```bash
docker compose logs nginx
# Error: bind() to 0.0.0.0:443 failed (98: Address already in use)
# or: duplicate "server_name yourdomain.com"
```

**Fix** — ensure nothing else is using port 80/443 (see Scenario B), then:

```bash
cd ~/app
docker compose down          # stop all containers
docker compose up -d         # start fresh
docker compose logs nginx    # confirm clean start
```

If the nginx config itself is wrong (e.g. stale `YOUR_DOMAIN` placeholder):

```bash
# Verify replacement was done
grep "server_name" ~/app/nginx/nginx.conf

# If it still shows YOUR_DOMAIN, fix it:
sed -i 's/YOUR_DOMAIN/yourdomain.com/g' ~/app/nginx/nginx.conf

# Restart nginx only
docker compose restart nginx
docker compose logs nginx --tail=20
```

---

### Quick Checklist for Domain Reset

Run through this before starting the main guide:

- [ ] DNS A record updated to new VPS IP
- [ ] `dig yourdomain.com +short` returns new VPS IP
- [ ] Nothing is running on port 80 (`sudo ss -tlnp | grep :80`)
- [ ] Nothing is running on port 443 (`sudo ss -tlnp | grep :443`)
- [ ] Certbot status checked (`sudo certbot certificates`)
- [ ] SSL certificate valid or freshly issued
- [ ] `~/app/nginx/nginx.conf` has actual domain name, not `YOUR_DOMAIN`

---

*Suno Discord Analysis — Retrieval-Web-Refactor*
