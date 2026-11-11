#!/usr/bin/env bash
# Fetch only the Apptronik Apollo model from Google DeepMind's MuJoCo Menagerie (Apache-2.0).
set -euo pipefail
DEST="${OBJECTMODEL_MENAGERIE:-$HOME/.cache/objectmodel/mujoco_menagerie}"
if [ -f "$DEST/apptronik_apollo/apptronik_apollo.xml" ]; then
  echo "already present: $DEST"; exit 0
fi
mkdir -p "$(dirname "$DEST")"
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set apptronik_apollo
echo "Apollo model at $DEST/apptronik_apollo (set OBJECTMODEL_MENAGERIE=$DEST if you used a custom path)"
