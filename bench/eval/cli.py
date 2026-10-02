"""Command line entry points.

    python -m bench.eval.cli evaluate --data DATASET_DIR --split test --pred PRED_DIR \
        [--val-pred VAL_PRED_DIR] --out runs/<run>/metrics/<dataset>_test
    python -m bench.eval.cli compare --a A.npz --b B.npz [--threshold fixed]
"""

import argparse
import json

import numpy as np

from bench.eval import stats
from bench.eval.evaluate import DEFAULT_PROTOCOL, evaluate, load_protocol


def _evaluate(args):
    proto = load_protocol(
        args.protocol,
        use_fov=None if args.use_fov is None else args.use_fov == "yes",
        label_policy=args.label_policy,
        lesion_metrics=False if args.no_lesion else None,
    )
    res = evaluate(args.data, args.split, args.pred, proto, args.val_pred, args.val_split, args.out)
    print(json.dumps(res["summary"], indent=1))


def _compare(args):
    a, b = np.load(args.a), np.load(args.b)
    if not np.array_equal(a["image_ids"], b["image_ids"]):
        raise SystemExit("the two evaluations were not run on the same images")
    classes = sorted({k.split("_")[0] for k in a.files if k.endswith("_pos")})
    out = {}
    for c in classes:
        boot = stats.paired_bootstrap_aupr(a[f"{c}_pos"], a[f"{c}_neg"], b[f"{c}_pos"], b[f"{c}_neg"])
        wil = stats.paired_wilcoxon(a[f"{c}_dice_{args.threshold}"], b[f"{c}_dice_{args.threshold}"])
        out[c] = {"aupr_paired_bootstrap": boot, "per_image_dice_wilcoxon": wil}
    p_adj = stats.holm([out[c]["per_image_dice_wilcoxon"]["p"] for c in classes])
    for c, p in zip(classes, p_adj):
        out[c]["per_image_dice_wilcoxon"]["p_holm"] = float(p)
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench.eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--pred", required=True)
    e.add_argument("--val-pred")
    e.add_argument("--val-split", default="val")
    e.add_argument("--out", required=True, help="output prefix; writes .json and .npz")
    e.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    e.add_argument("--use-fov", choices=["yes", "no"])
    e.add_argument("--label-policy", choices=["multilabel", "m2mrf_overwrite"])
    e.add_argument("--no-lesion", action="store_true")
    e.set_defaults(func=_evaluate)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--threshold", default="fixed")
    c.set_defaults(func=_compare)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
