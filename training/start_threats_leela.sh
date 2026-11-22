#!/bin/bash
# threats-Netz nur mit externen Daten (Leela test80 u. a., BT4-relabelt, Stockfish-Binpacks aus
# vondele/nettest, siehe download_leela_data.sh). Filter und Score-Ziel wie nnue-pytorch/nettest
# (sf_loader.rs), WDL-Zyklus 0 -> 0.3 -> 0 wie dort (lambda-cycle).
#
# Start:   ./start_threats_leela.sh            (SUPERBATCHES=1600 NO_FINETUNE=1 voreingestellt)
# Endnetz: stage1-1165 (ohne Finetune; bei 1600: 145 Warmup + 1165 Hauptstufe = 1309 SBs, ~14 h)
# quantised_threats.bin im Repo bleibt unverändert (INSTALL_NET=0), nur die Variante bekommt das Netz.
# Startet keinen SPRT.
# Fortsetzen: RESUME_ARGS="--start-stage 1 --start 171 -r checkpoints/sleepmind_threats_leela-stage1-170/optimiser_state" ./start_threats_leela.sh
set -euo pipefail

ROOT=/home/paschty/workspace/sleepmind2
DATA=${DATA:-$ROOT/training_data/leela}
VARIANT=${VARIANT:-threats_leela}
SUPERBATCHES=${SUPERBATCHES:-1600}
NO_FINETUNE=${NO_FINETUNE:-1}
RESUME_ARGS=${RESUME_ARGS:-}
LOG=$ROOT/training/pipeline_$VARIANT.log

if systemctl --user is-active --quiet leela-download; then echo "FEHLER: Download läuft noch"; exit 1; fi
if systemctl --user is-active --quiet threats-pipeline; then echo "FEHLER: threats-pipeline läuft schon"; exit 1; fi
[ ! -e "$ROOT/variants/$VARIANT" ] || { echo "FEHLER: Variante $VARIANT existiert schon"; exit 1; }

DATA_ARGS=""
for f in "$DATA"/*.binpack; do
    [ -f "$f" ] || { echo "FEHLER: keine .binpack-Dateien in $DATA"; exit 1; }
    DATA_ARGS="$DATA_ARGS --sf $f"
done
echo "Daten: $(ls "$DATA"/*.binpack | wc -l) Dateien, $(du -sh "$DATA" | cut -f1)"

cd "$ROOT/training"
systemd-run --user --unit threats-pipeline --working-directory="$PWD" --setenv=PATH="$PATH" \
    --setenv=VARIANT="$VARIANT" --setenv=SUPERBATCHES="$SUPERBATCHES" --setenv=NO_FINETUNE="$NO_FINETUNE" \
    --setenv=INSTALL_NET=0 --setenv=DATA_ARGS="$DATA_ARGS" --setenv=RESUME_ARGS="$RESUME_ARGS" \
    -p StandardOutput=append:"$LOG" -p StandardError=append:"$LOG" ./run_pipeline_threats.sh

echo
echo "Gestartet. Fortschritt:"
echo "  tail -f $LOG"
echo "  cat $ROOT/variants/${VARIANT}_pipeline.status"
