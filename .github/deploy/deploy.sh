#!/usr/bin/env bash
#
# inference-provider-proxy deploy script (runs on the server as root).
#
# The release workflow renders this template -- it substitutes {{VERSION}},
# {{UPSTREAM_URL}} and {{UPSTREAM_KEY}} with the release tag and repository
# secret values -- then pushes the result to the server and executes it.
#
# The script is idempotent: the first run bootstraps the machine (uv, .env,
# systemd unit, firewall), later runs only swap the wheel and restart.

set -Eeuo pipefail

VERSION='{{VERSION}}'
UPSTREAM_URL='{{UPSTREAM_URL}}'
UPSTREAM_KEY='{{UPSTREAM_KEY}}'

APP_DIR=/opt/ipp
PORT=13306
SERVICE=ipp.service
WHEEL_URL="https://github.com/JFortun/inference-provider-proxy/releases/download/v${VERSION}/inference_provider_proxy-${VERSION}-py3-none-any.whl"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

wheel=""
stopped=0
on_error() {
  if [ -n "${wheel}" ]; then
    rm -f "${wheel}" || true
  fi
  if [ "${stopped}" -eq 1 ]; then
    log "deploy failed, restarting the previous service"
    systemctl start "${SERVICE}" || true
  fi
}
trap on_error ERR

# --- uv and Python ----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "uv not found, installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:/usr/local/bin:${PATH}"
log "uv $(uv --version | awk '{print $2}')"
log "ensuring Python 3.15 is available"
uv python install 3.15

# --- configuration (.env: created once, never overwritten) ------------------
mkdir -p "${APP_DIR}"
if [ ! -f "${APP_DIR}/.env" ]; then
  log "creating ${APP_DIR}/.env"
  cat >"${APP_DIR}/.env" <<EOF
IPP_UPSTREAM_BASE_URL=${UPSTREAM_URL}
IPP_UPSTREAM_API_KEY=${UPSTREAM_KEY}
IPP_BIND_HOST=0.0.0.0
IPP_BIND_PORT=${PORT}
EOF
fi
chmod 600 "${APP_DIR}/.env"

# --- swap the wheel (stop first so the old venv is never yanked mid-run) ----
if systemctl is-active --quiet "${SERVICE}"; then
  log "stopping ${SERVICE} for the upgrade"
  systemctl stop "${SERVICE}"
  stopped=1
fi

log "installing inference-provider-proxy ${VERSION}"
wheel="/tmp/ipp-wheel-${VERSION}.whl"
curl -fsSL "${WHEEL_URL}" -o "${wheel}"
uv tool install --force "${wheel}"
rm -f "${wheel}"
wheel=""

IPP_BIN="$(command -v ipp || true)"
if [ -z "${IPP_BIN}" ]; then
  log "ipp entry point not found on PATH after install"
  exit 1
fi

# --- systemd unit (installed when missing, refreshed when this template differs)
new_unit="$(mktemp)"
cat >"${new_unit}" <<EOF
[Unit]
Description=inference-provider-proxy
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
ExecStart=${IPP_BIN}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
if ! cmp -s "${new_unit}" "/etc/systemd/system/${SERVICE}"; then
  log "installing systemd unit ${SERVICE}"
  install -m 644 "${new_unit}" "/etc/systemd/system/${SERVICE}"
  systemctl daemon-reload
fi
rm -f "${new_unit}"

# --- firewall (only when ufw is actually in use) ----------------------------
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  log "allowing ${PORT}/tcp through ufw"
  ufw allow "${PORT}/tcp" >/dev/null || true
fi

# --- start and health check -------------------------------------------------
log "starting ${SERVICE}"
systemctl enable "${SERVICE}" >/dev/null 2>&1 || true
systemctl restart "${SERVICE}"
stopped=0

for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    log "healthy on port ${PORT}"
    printf 'DEPLOY OK %s\n' "${VERSION}"
    exit 0
  fi
  sleep 1
done

log "service did not become healthy"
systemctl --no-pager -l status "${SERVICE}" || true
journalctl -u "${SERVICE}" -n 40 --no-pager || true
exit 1
