#!/usr/bin/env bash
# Build the driver image from this repository. Run it once, on the Linux
# computer that will sit on the same Wi-Fi as the dog:
#   ./controller/docker/build.sh
# The image is tagged go2:dev. Override with GO2_IMAGE=myname ./controller/docker/build.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
IMAGE=${GO2_IMAGE:-go2:dev}
echo "Building $IMAGE from $ROOT"
docker build -f "$ROOT/controller/docker/Dockerfile" -t "$IMAGE" "$ROOT"
echo "Built $IMAGE. Next: connect this computer to the lab Wi-Fi and run ./start_console.sh"
