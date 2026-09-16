#!/bin/bash
# Concatenate a per-structure POTCAR from a local VASP PBE pseudopotential
# distribution. POTCAR files are licensed with VASP and are not distributed
# in this repo -- set POTCAR_DIR to your own installation before running.
#
# Usage: POTCAR_DIR=/path/to/potpaw_PBE ./generate_potcar.sh Cu O
set -euo pipefail

POTCAR_DIR="${POTCAR_DIR:?Set POTCAR_DIR to your local VASP PBE pseudopotential directory}"

> POTCAR
for element in "$@"; do
    if [ -f "$POTCAR_DIR/$element/POTCAR" ]; then
        cat "$POTCAR_DIR/$element/POTCAR" >> POTCAR
        echo "Added $element from $POTCAR_DIR/$element/POTCAR"
    elif [ -f "$POTCAR_DIR/${element}_sv/POTCAR" ]; then
        cat "$POTCAR_DIR/${element}_sv/POTCAR" >> POTCAR
        echo "Added ${element}_sv from $POTCAR_DIR/${element}_sv/POTCAR"
    else
        echo "WARNING: POTCAR for $element not found (also tried ${element}_sv)" >&2
        exit 1
    fi
done

echo "POTCAR generation completed successfully"
