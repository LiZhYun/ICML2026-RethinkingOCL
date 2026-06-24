"""Build a per-job 100K-schedule snippet by merging a variant snippet with
the 100K schedule overrides (max_steps, all param_group decay_steps).

Resolves the dataset's per-group LRs from the dataset rescue config so the
output snippet is self-contained and can be passed to train.py via
--config_overrides_file.

Usage:
    python3 scripts/build_100k_snippet.py \\
        --base configs/slotcontrast/app/movi_c_slotcontrast_rescue15k.yaml \\
        --variant configs/slotcontrast/snippets/softident_st_fr15.yaml \\
        --out /tmp/100k/movic_sc_st_fr15.yaml

The output snippet contains:
  - All keys from the variant snippet (predictor, loss_weights, etc.)
  - trainer.max_steps = 100000
  - optimizer.param_groups[*].lr_scheduler.decay_steps = 100000 (preserving lrs)
  - For frozen variants (param_groups: null in variant), the global lr_scheduler
    is set to decay_steps=100000.
"""
import argparse
import os
from omegaconf import OmegaConf


def build(base_path, variant_path, out_path):
    base = OmegaConf.load(base_path)
    variant = OmegaConf.load(variant_path)
    # Merge variant onto base to see if param_groups is null (frozen) or list
    merged_for_inspection = OmegaConf.merge(base, variant)
    is_frozen = merged_for_inspection.optimizer.get("param_groups") is None

    # Start the output snippet from the variant snippet (preserving its keys)
    out = OmegaConf.create(OmegaConf.to_container(variant, resolve=False))

    # Always set max_steps=100000
    if "trainer" not in out:
        out["trainer"] = {}
    out["trainer"]["max_steps"] = 100000

    if is_frozen:
        # Frozen baseline: variant should already have param_groups: null and a global lr_scheduler.
        # Just update decay_steps in the global scheduler.
        if "optimizer" not in out:
            out["optimizer"] = {}
        out["optimizer"]["param_groups"] = None
        out["optimizer"]["lr_scheduler"] = {
            "name": "exp_decay_with_warmup",
            "warmup_steps": 2500,
            "decay_steps": 100000,
        }
    else:
        # Non-frozen: rebuild param_groups with decay_steps=100000 but preserve lrs/include/exclude.
        # Prefer variant's param_groups if it overrode them (e.g. full-FT drops the
        # LoRA group); else inspect the BASE config's param_groups for the LR structure.
        variant_pgs = OmegaConf.select(variant, "optimizer.param_groups", default=None)
        if variant_pgs is not None and len(variant_pgs) > 0:
            pg_source = variant_pgs
        else:
            pg_source = base.optimizer.param_groups
        new_pgs = []
        for pg in pg_source:
            entry = OmegaConf.to_container(pg, resolve=True)
            # Update decay_steps; preserve everything else.
            entry["lr_scheduler"]["decay_steps"] = 100000
            new_pgs.append(entry)
        if "optimizer" not in out:
            out["optimizer"] = {}
        out["optimizer"]["param_groups"] = new_pgs

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(OmegaConf.to_yaml(out))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = build(args.base, args.variant, args.out)
    print(p)
