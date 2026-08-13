#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One-command local WebArena stack: load the 4 official images and start
# 3 replicas per site on the ports experiments/webarena expects
# (CONTAINERS / URLS in dispatcher.sh).
#
# Prereqs: docker access; the 4 official tars downloaded (see IMAGES below;
#          sources: http://metis.lti.cs.cmu.edu/webarena-images/ or the
#          mirrors listed in webarena's environment_docker README).
#
# Usage:
#   IMAGE_DIR=~/webarena_images bash experiments/webarena/setup_stack.sh
#   SITES="reddit gitlab" bash ...    # subset
#
# Idempotent: running containers are left alone; missing ones are created.
# GitLab replicas take ~3-5 min each to pass health (reconfigure + boot).
set -euo pipefail

IMAGE_DIR=${IMAGE_DIR:-$HOME/webarena_images}
SITES=${SITES:-"reddit shopping shopping_admin gitlab"}
HOSTNAME_FOR_URLS=${HOSTNAME_FOR_URLS:-127.0.0.1}

declare -A TAR IMG
TAR[shopping]=shopping_final_0712.tar;                IMG[shopping]=shopping_final_0712
TAR[shopping_admin]=shopping_admin_final_0719.tar;    IMG[shopping_admin]=shopping_admin_final_0719
TAR[reddit]=postmill-populated-exposed-withimg.tar;   IMG[reddit]=postmill-populated-exposed-withimg
TAR[gitlab]=gitlab-populated-final-port8023.tar;      IMG[gitlab]=gitlab-populated-final-port8023

# site -> "name:hostport" x3 (must match CONTAINERS / URLS in dispatcher.sh)
declare -A REPLICAS CONTAINER_PORT
REPLICAS[reddit]="forum:19999 forum_1:19998 forum_2:19997";                     CONTAINER_PORT[reddit]=80
REPLICAS[shopping]="shopping:17770 shopping_1:17769 shopping_2:17768";          CONTAINER_PORT[shopping]=80
REPLICAS[shopping_admin]="shopping_admin:17780 shopping_admin_1:17779 shopping_admin_2:17778"; CONTAINER_PORT[shopping_admin]=80
REPLICAS[gitlab]="gitlab:18023 gitlab_1:18022 gitlab_2:18021";                  CONTAINER_PORT[gitlab]=8023

log() { echo "[setup_stack $(date +%H:%M:%S)] $*"; }

load_image() {  # $1 site
    local img=${IMG[$1]} tar=$IMAGE_DIR/${TAR[$1]}
    if docker image inspect "$img" >/dev/null 2>&1; then
        log "image $img already loaded"; return
    fi
    [ -f "$tar" ] || { echo "missing $tar -- download it first"; exit 1; }
    log "docker load $tar (this can take many minutes)"
    docker load --input "$tar"
}

wait_http() {  # $1 url, $2 timeout_s -- logs TIMEOUT but never fails the
    # whole script (set -e): a slow replica shouldn't abort the others.
    local t=0
    until curl -sf -o /dev/null --max-time 5 "$1"; do
        t=$((t + 5)); [ "$t" -ge "$2" ] && { log "TIMEOUT waiting for $1"; return 0; }
        sleep 5
    done
}

start_replica() {  # $1 site, $2 name, $3 hostport
    local site=$1 name=$2 port=$3 img=${IMG[$1]} cport=${CONTAINER_PORT[$1]}
    if docker ps --format '{{.Names}}' | grep -qx "$name"; then
        log "$name already running"; return
    fi
    docker rm -f "$name" >/dev/null 2>&1 || true
    if [ "$site" = gitlab ]; then
        # external_url's port is where gitlab LISTENS inside the container
        # after reconfigure -- so host and container port must be the SAME
        # (mapping host:$port -> container:8023 while external_url says
        # :$port leaves nothing listening on 8023 => connection refused).
        # --shm-size: docker's 64MB default breaks gitlab's prometheus
        # mmap files (IOError "unmapped file" -> flapping 500s on every
        # page). We also disable prometheus monitoring outright -- eval
        # doesn't need it and it's the sole consumer of those mmaps.
        docker run --name "$name" -d -p "$port:$port" --shm-size 1g "$img" \
            /opt/gitlab/embedded/bin/runsvdir-start
        docker exec "$name" sed -i \
            "s|^external_url.*|external_url 'http://${HOSTNAME_FOR_URLS}:${port}'|" \
            /etc/gitlab/gitlab.rb
        # Pin worker sizes: omnibus reconfigure auto-scales puma/sidekiq
        # from HOST cpu count (bad on many-core hosts) and blows past
        # postgres's connection slots ("remaining connection slots are
        # reserved...").
        # Each replica serves 1-3 browser sessions; 4 puma workers suffice.
        docker exec "$name" bash -c "cat >> /etc/gitlab/gitlab.rb <<'RB'
prometheus_monitoring['enable'] = false
puma['worker_processes'] = 4
sidekiq['max_concurrency'] = 10
postgresql['max_connections'] = 300
RB"
        docker exec "$name" gitlab-ctl reconfigure >/dev/null
        wait_http "http://${HOSTNAME_FOR_URLS}:${port}/users/sign_in" 1200
    else
        docker run --name "$name" -d -p "$port:$cport" "$img"
        if [ "$site" != reddit ]; then   # magento sites need per-port base_url
            local base="http://${HOSTNAME_FOR_URLS}:${port}"
            [ "$site" = shopping_admin ] || true
            sleep 10   # let apache/mysql come up before magento CLI
            docker exec "$name" /var/www/magento2/bin/magento \
                setup:store-config:set --base-url="$base" >/dev/null
            docker exec "$name" mysql -u magentouser -pMyPassword magentodb \
                -e "UPDATE core_config_data SET value='$base/' WHERE path='web/secure/base_url';" \
                >/dev/null 2>&1 || true
            docker exec "$name" /var/www/magento2/bin/magento cache:flush >/dev/null
        fi
        wait_http "http://${HOSTNAME_FOR_URLS}:${port}" 300
    fi
    log "$name healthy on :$port"
}

for site in $SITES; do
    load_image "$site"
done
for site in $SITES; do
    for pair in ${REPLICAS[$site]}; do
        start_replica "$site" "${pair%%:*}" "${pair##*:}"
    done
done
log "stack ready: $SITES"
