#!/usr/bin/env bash
set -euo pipefail

# Setup this app on a plain Ubuntu/Debian VPS without Docker.
# - Builds frontend and serves via Nginx
# - Runs backend (FastAPI/uvicorn) as a systemd service
#
# Usage:
#   sudo bash scripts/setup-vps.sh <domain> [letsencrypt-email]
# Example:
#   sudo bash scripts/setup-vps.sh whyisthestockplummeting.com you@example.com

DOMAIN=${1:-}
LE_EMAIL=${2:-}

if [[ -z "$DOMAIN" ]]; then
  echo "Usage: $0 <domain> [letsencrypt-email]" >&2
  exit 1
fi

APP_NAME="whyisthestockplummeting"
APP_DIR="/opt/${APP_NAME}"
FRONTEND_BUILD_DIR="${APP_DIR}/frontend/dist"
FRONTEND_WEB_ROOT="/var/www/${APP_NAME}"
ENV_FILE="/etc/${APP_NAME}.env"
SERVICE_FILE="/etc/systemd/system/${APP_NAME}-backend.service"
NGINX_SITE="/etc/nginx/sites-available/${APP_NAME}.conf"
NGINX_SITE_LINK="/etc/nginx/sites-enabled/${APP_NAME}.conf"
DATA_DIR="/var/lib/${APP_NAME}"

echo "--- Installing system packages ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-pip nginx curl gnupg ca-certificates lsb-release \
  fonts-noto-core fonts-ubuntu fontconfig

if ! command -v node >/dev/null 2>&1; then
  echo "--- Installing Node.js 20.x ---"
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

echo "--- Creating app directory at ${APP_DIR} ---"
mkdir -p "$APP_DIR"
rsync -a --delete --exclude ".git" ./ "$APP_DIR"/

echo "--- Setting up Python virtualenv ---"
cd "$APP_DIR/backend"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "--- Building frontend ---"
cd "$APP_DIR/frontend"
npm ci
npm run build

echo "--- Installing frontend to ${FRONTEND_WEB_ROOT} ---"
mkdir -p "$FRONTEND_WEB_ROOT"
rsync -a --delete "$FRONTEND_BUILD_DIR"/ "$FRONTEND_WEB_ROOT"/
chown -R www-data:www-data "$FRONTEND_WEB_ROOT"

echo "--- Preparing application data dir at ${DATA_DIR} ---"
mkdir -p "${DATA_DIR}/news"
chown -R www-data:www-data "${DATA_DIR}"

echo "--- Writing environment file at ${ENV_FILE} ---"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"
if ! grep -q "^PUBLIC_WEB_ORIGIN=" "$ENV_FILE" 2>/dev/null; then
  echo "PUBLIC_WEB_ORIGIN=https://${DOMAIN}" >> "$ENV_FILE"
fi
if ! grep -q "^OPENAI_API_KEY=" "$ENV_FILE" 2>/dev/null; then
  echo "OPENAI_API_KEY=CHANGE_ME" >> "$ENV_FILE"
fi

# Persist SQLite DB under /var/lib with proper permissions
if ! grep -q "^SQLITE_DB_PATH=" "$ENV_FILE" 2>/dev/null; then
  echo "SQLITE_DB_PATH=${DATA_DIR}/stockdrop.db" >> "$ENV_FILE"
fi

# Persist news cache outside code directory
if ! grep -q "^NEWS_CACHE_DIR=" "$ENV_FILE" 2>/dev/null; then
  echo "NEWS_CACHE_DIR=${DATA_DIR}/news" >> "$ENV_FILE"
fi

# Prefer a known, scalable font for OG image rendering
if ! grep -q "^OG_FONT_PATH=" "$ENV_FILE" 2>/dev/null; then
  if [[ -f "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf" ]]; then
    echo "OG_FONT_PATH=/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf" >> "$ENV_FILE"
  elif [[ -f "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf" ]]; then
    echo "OG_FONT_PATH=/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf" >> "$ENV_FILE"
  fi
fi

# Rebuild font cache so newly installed fonts are discoverable
fc-cache -f >/dev/null 2>&1 || true

echo "--- Creating systemd service ---"
cat > "$SERVICE_FILE" <<SYSTEMD
[Unit]
Description=${APP_NAME} backend (FastAPI)
After=network.target

[Service]
Type=simple
EnvironmentFile=${ENV_FILE}
WorkingDirectory=${APP_DIR}/backend
ExecStart=${APP_DIR}/backend/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
User=www-data
Group=www-data

[Install]
WantedBy=multi-user.target
SYSTEMD

systemctl daemon-reload
systemctl enable ${APP_NAME}-backend
systemctl restart ${APP_NAME}-backend

echo "--- Configuring Nginx ---"
cat > "$NGINX_SITE" <<NGINX
server {
    listen 8080;
    server_name ${DOMAIN};

    # Serve static frontend
    root ${FRONTEND_WEB_ROOT};
    index index.html;

    # Security & performance
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml+rss text/javascript;

    # API proxy to FastAPI backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Single Page App fallback
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
NGINX

ln -sf "$NGINX_SITE" "$NGINX_SITE_LINK"
nginx -t
systemctl reload nginx

# Open firewall port 8080 and close 80 if UFW is available
if command -v ufw >/dev/null 2>&1; then
  ufw allow 8080/tcp || true
  ufw delete allow 80/tcp || true
  ufw deny 80/tcp || true
fi

if [[ -n "$LE_EMAIL" ]]; then
  if ! command -v certbot >/dev/null 2>&1; then
    echo "--- Installing certbot (Let's Encrypt) ---"
    apt-get install -y certbot python3-certbot-nginx
  fi
  echo "--- Obtaining TLS certificate for ${DOMAIN} ---"
  certbot --nginx -n --agree-tos -m "$LE_EMAIL" -d "$DOMAIN" || true
fi

echo "--- Done ---"
echo "Next steps:"
echo "1) Edit ${ENV_FILE} and set a real OPENAI_API_KEY"
echo "2) systemctl restart ${APP_NAME}-backend"
echo "3) Visit: http://${DOMAIN}:8080"
