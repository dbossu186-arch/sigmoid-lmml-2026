#!/usr/bin/env bash
# fetch a run's student from the training box, pack at a given tau and run the grader-style check
# usage: scripts/fetch_eval.sh <run_name> <tau> <host>
set -e
run=$1; tau=$2; host=$3
mkdir -p runs/$run
scp -q $host:vietnam/runs/$run/student.pth runs/$run/
cp runs/d_ho15_feat/holdout_idx.npy runs/$run/holdout_idx.npy
PY=${PY:-python}
$PY scripts/pack_cnx.py runs/$run/student.pth --tau $tau --out runs/$run/model.pt
$PY scripts/verify_cnx.py runs/$run/model.pt runs/$run/holdout_idx.npy
