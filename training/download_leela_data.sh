#!/bin/bash
# Lädt die externen Stockfish-Trainingsdaten (Leela-Partien, mit BT4 neu bewertet) von
# Hugging Face, Auswahl aus vondele/nettest threats.yaml, ~195 GB:
#   test80 2023 (alle 12 Monate) + 2024 jan/feb, test80 2022 nov,
#   T60T70wIsRightFarseer split_0, tb5dtm (Tablebase-Endspiele)
# Nicht genommen: sf-searchtree (keine Partieergebnisse, ply 9999), dfrc (Chess960),
# die .q.binpack-Varianten und die älteren test60/77/78/79-Monate.
# Fortsetzbar: bereits vollständige Dateien werden übersprungen, halbe mit curl -C - fortgesetzt.
# Als systemd-User-Unit starten:
#   systemd-run --user --unit leela-download -p StandardOutput=append:$PWD/download_leela.log \
#       -p StandardError=append:$PWD/download_leela.log ./download_leela_data.sh
set -euo pipefail

DEST=${DEST:-/home/paschty/workspace/sleepmind2/training_data/leela}
mkdir -p "$DEST"

FILES=(
    vondele/linrock_relabel_2/test80-2023-01-jan-16tb7p.v6-sk20.min.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-02-feb-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-03-mar-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-04-apr-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-05-may-2tb7p.v6.min.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-06-jun-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-07-jul-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-08-aug-2tb7p.v6.min.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-09-sep-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-11-nov-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_2/test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    xushawn/test80-bt4-relabel/test80-2024-01-jan-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    xushawn/test80-bt4-relabel/test80-2024-02-feb-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack
    vondele/linrock_relabel_1/test80-2022-11-nov-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack
    vondele/from_kaggle_2_relabel/T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack
    vondele/rescored/tb5dtm.binpack
)

for entry in "${FILES[@]}"; do
    repo=${entry%/*}
    file=${entry##*/}
    out="$DEST/$file"
    size=$(curl -sL "https://huggingface.co/api/datasets/$repo/paths-info/main" -d "paths=$file" \
        | python3 -I -c "import json,sys; e=json.load(sys.stdin)[0]; print(e.get('lfs',{}).get('size',e['size']))")
    if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
        echo "$(date '+%F %T') vorhanden: $file"
        continue
    fi
    echo "$(date '+%F %T') lade $file ($((size / 1000000)) MB)"
    curl -fL --retry 20 --retry-delay 30 --retry-all-errors -C - -sS -o "$out" \
        "https://huggingface.co/datasets/$repo/resolve/main/$file"
    got=$(stat -c %s "$out")
    [ "$got" = "$size" ] || { echo "FEHLER: $file hat $got statt $size Bytes"; exit 1; }
    echo "$(date '+%F %T') fertig: $file"
done
echo "$(date '+%F %T') alle Dateien vollständig in $DEST"
du -sh "$DEST"
