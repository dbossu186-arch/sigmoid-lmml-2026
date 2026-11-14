#!/usr/bin/env bash
# Orientation-aware PRNU pipeline
set -e
cd ~/sigmoid_japan
export NTHREADS=2 PRNU_SIGMA=3.0
venv/bin/python -I cache_feats.py dataset cache_wav wav
venv/bin/python -I orient_attr.py cache_wav out_orient --thr 15 --rounds 4
venv/bin/python -I joint_em.py cache_wav out_orient/orient_attrib.json out_joint --thr 15 --rounds 5
venv/bin/python -I substructure.py cache_wav out_joint/joint.json toorisugari > out_joint/sub_toorisugari.txt
echo PIPELINE_DONE
