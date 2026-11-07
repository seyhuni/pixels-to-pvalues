"""Run a whole experiment plan (sequence of experiments) - resume-safe.

python scripts/run_plan.py plans/stage1_backbones.yaml
python scripts/run_plan.py plans/stage2_ablation.yaml --set model.encoder=fm:uni2h
python scripts/run_plan.py plans/stage1_backbones.yaml --dry

A plan file:
  base: configs/base.yaml
  common: [train.epochs=60]                # overrides applied to every run
  runs:
    - {name: s1_cnn, exp: configs/exp/main.yaml, set: [model.encoder=cnn:convnext_small.fb_in22k]}
Failures are logged to results/plan_failures.log and the plan continues.
"""
import argparse
import datetime
import sys
import traceback

import yaml

import _common
from frpv.config import load_config, resolve_path
from frpv.train import collect_oof, run_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides for all runs")
    ap.add_argument("--only", nargs="*", default=None, help="run only these run names")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    plan = yaml.safe_load(open(resolve_path(a.plan), encoding="utf-8"))
    base = plan.get("base", "configs/base.yaml")
    common = plan.get("common", []) + a.set
    log = _common.results_dir() / "plan_failures.log"
    n_fail = 0
    for run in plan["runs"]:
        if a.only and run["name"] not in a.only:
            continue
        ov = common + run.get("set", [])
        cfg = load_config(base, run.get("exp"), overrides=ov)
        name = run["name"]
        if plan.get("tag_encoder"):
            from pick_best import enc_tag
            name = f"{name}_{enc_tag(cfg.model.encoder)}"
        cfg = load_config(base, run.get("exp"), overrides=ov + [f"exp_name={name}"])
        folds = list(cfg.data.folds)
        if cfg.data.split == "lopo":
            folds = [f for f in folds if f < 4]
        print(f"\n######## {name}: {cfg.model.encoder} | {cfg.model.head} | {cfg.model.label_mode} | "
              f"aug={cfg.stain.augment} branch={cfg.stain.branch_input} | {cfg.data.split} {folds}")
        if a.dry:
            continue
        for k in folds:
            try:
                run_fold(cfg, k)
            except Exception:
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, "a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.datetime.now()}] {name} fold {k}\n{traceback.format_exc()}")
                print(f"!! {name} fold {k} failed - see {log}")
                n_fail += 1
        collect_oof(cfg)
    if n_fail:
        print(f"{n_fail} fold(s) failed - see {log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
