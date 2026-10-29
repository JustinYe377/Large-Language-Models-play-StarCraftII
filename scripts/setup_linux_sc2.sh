#!/usr/bin/env bash
# Install the headless Linux StarCraft II build and this repo's ladder maps.
#
#   scripts/setup_linux_sc2.sh --accept-eula            # installs into ~/StarCraftII
#   SC2PATH=/data/StarCraftII scripts/setup_linux_sc2.sh --accept-eula
#
# Downloading the Linux build means accepting Blizzard's AI and Machine Learning
# License (https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html),
# so the download only happens with --accept-eula. Safe to re-run: an existing
# install is kept and only missing maps are copied.
set -euo pipefail

SC2_VERSION="4.10"
SC2_ZIP_URL="https://blzdistsc2-a.akamaihd.net/Linux/SC2.${SC2_VERSION}.zip"
SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCEPT_EULA=0
[[ "${1:-}" == "--accept-eula" ]] && ACCEPT_EULA=1

if compgen -G "$SC2PATH/Versions/Base*/SC2_x64" > /dev/null; then
    echo "SC2 already installed: $(ls -d "$SC2PATH"/Versions/Base* | tail -1)"
else
    if [[ $ACCEPT_EULA -ne 1 ]]; then
        echo "No Linux SC2 build in $SC2PATH."
        echo "Re-run with --accept-eula to download SC2 $SC2_VERSION (accepts Blizzard's AI/ML license)."
        exit 1
    fi
    parent="$(dirname "$SC2PATH")"
    zip="$parent/SC2.${SC2_VERSION}.zip"
    mkdir -p "$parent"
    [[ -f "$zip" ]] || { echo "Downloading $SC2_ZIP_URL (~4 GB)"; curl -fL -o "$zip" "$SC2_ZIP_URL"; }
    echo "Unpacking into $parent"
    unzip -q -P iagreetotheeula -o "$zip" -d "$parent"
    if [[ "$parent/StarCraftII" != "$SC2PATH" ]]; then
        mv "$parent/StarCraftII" "$SC2PATH"
    fi
fi

mkdir -p "$SC2PATH/Maps"
copied=0
for map in "$REPO_ROOT"/Maps/*.SC2Map; do
    dest="$SC2PATH/Maps/$(basename "$map")"
    if [[ ! -f "$dest" ]]; then
        cp "$map" "$dest"
        copied=$((copied + 1))
    fi
done
echo "Maps: copied $copied new, $(ls "$SC2PATH"/Maps/*.SC2Map | wc -l) ladder maps in $SC2PATH/Maps"
echo
echo "Run headless with:  --sc2_platform Linux --sc2_path $SC2PATH"
