#!/bin/bash
# Wartet auf das Ende der Unit threats-pipeline (Leela-Training, start_threats_leela.sh) und
# startet dann nacheinander zwei SPRTs threats_leela vs. main_91bcc74 (aktueller main):
#   1. feste Knoten (30k)   2. LTC 60+0.6, nur wenn der Fixed-Nodes-SPRT H1 annimmt
# Als systemd-User-Unit starten:
#   systemd-run --user --unit leela-sprt --working-directory=$PWD --setenv=PATH=$PATH ./run_sprt_after_leela.sh
set -euo pipefail

ROOT=/home/paschty/workspace/sleepmind2
VARIANT=${VARIANT:-threats_leela}
BASE=${BASE:-main_91bcc74}
PIPE_UNIT=${PIPE_UNIT:-threats-pipeline}
PIPE_STATUS="$ROOT/variants/${VARIANT}_pipeline.status"
STATUS="$ROOT/variants/${VARIANT}_sprt.status"
NODES=${NODES:-30000}
LTC=${LTC:-60+0.6}
CONCURRENCY=${CONCURRENCY:-16}

status() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
trap 'status "FEHLER in Zeile $LINENO (exit $?)"' ERR

status "Warte auf Unit $PIPE_UNIT"
sleep 60
while systemctl --user is-active --quiet "$PIPE_UNIT"; do sleep 60; done

if ! tail -n 1 "$PIPE_STATUS" | grep -q "Variante $VARIANT angelegt, Repo-Build wieder Legacy"; then
    status "Pipeline nicht erfolgreich beendet ($(tail -n 1 "$PIPE_STATUS")), kein SPRT"
    exit 1
fi

cd "$ROOT"
[ -x "variants/$VARIANT/sleepmind" ] || { status "FEHLER: variants/$VARIANT/sleepmind fehlt"; exit 1; }
[ -x "variants/$BASE/sleepmind" ] || { status "FEHLER: variants/$BASE/sleepmind fehlt"; exit 1; }

LOG="$ROOT/variants/${VARIANT}_vs_${BASE}_sprt_nodes${NODES}.log"
status "Start SPRT $VARIANT vs $BASE, nodes=$NODES, concurrency=$CONCURRENCY"
NODES=$NODES CONCURRENCY=$CONCURRENCY ./tournament.sh sprt "$VARIANT" "$BASE" > "$LOG" 2>&1 || true
status "Fixed-Nodes-SPRT beendet: $(grep -E '^(Elo|LLR)' "$LOG" | tail -n 2 | tr '\n' ' ')"
if ! grep -q "H1 was accepted" "$LOG"; then
    status "Fixed Nodes nicht besser (kein H1), kein LTC"
    exit 0
fi

LOG="$ROOT/variants/${VARIANT}_vs_${BASE}_sprt_ltc.log"
status "Start SPRT $VARIANT vs $BASE, tc=$LTC, concurrency=$CONCURRENCY"
TC=$LTC CONCURRENCY=$CONCURRENCY ./tournament.sh sprt "$VARIANT" "$BASE" > "$LOG" 2>&1 || true
status "LTC-SPRT beendet: $(grep -E '^(Elo|LLR)' "$LOG" | tail -n 2 | tr '\n' ' ')"
