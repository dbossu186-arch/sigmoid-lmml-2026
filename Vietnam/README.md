# That isn't my shoe! (open-set sneaker recognition)

Two parts.

1. Teacher: zero-shot MobileCLIP2-S0. Image embedding vs text prototypes for the
   3 known brands and ~75 negative prompts (other brands, "fake", "unbranded").
   Unknown if (best known cosine - best negative cosine) < -0.0262. No training.

2. Student: torchvision convnext_tiny (ImageNet weights), pruned to stage 0/1 full,
   5 blocks in stage 2, none in stage 3. Distilled from the teacher on the 735
   known-brand images only (heavy augmentation, patch shuffle, cutmix, random
   erasing; soft targets plus embedding matching).

Checkpoint format (model.pt): plain dict with classes, tau, mean, std,
state_dict. Dropped blocks have layer_scale 0 and stride-0 zero tensors. Kept
blocks store both pointwise linears as int8; the per-channel scale is folded into
layer_scale and the preceding LayerNorm, so a plain float() + strict load is exact.
Rule: max softmax < tau gives unknown.

Scripts:
- scripts/feats_e0.py, margin_e0.py: teacher embeddings and thresholds
- scripts/distill_cnx.py: student training
- scripts/pack_cnx.py: write the checkpoint
- scripts/verify_cnx.py: load like the grader and evaluate

Final model (model.pt): weight average of four student runs trained 150 epochs with
different recipes (brand-mark hiding from a teacher occlusion map, Caltech101 extra
inputs, EMA). tau = 0.61. On the test set: known accuracy 0.856, rejection 0.700.

first_attempt/ holds an earlier TorchScript export of the teacher. The platform loads
checkpoints with weights_only=True, so that format was not accepted.
