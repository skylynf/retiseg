"""E5 inference tiers (design/15 sections 4-6, scope docs/paper/16 section 3, tasks docs/ops/17 section 6).

    python -m bench.e5 prompts   --set IDRiD [--split test]                      oracle prompts from the labels (once per set)
    python -m bench.e5 areacap                                                   training-set area cap for e5_frozen.yaml
    python -m bench.e5 amg       --model sam|sam2 --generator default|more_masks --set DDR [--split val|test] [--geometry G1|G2] [--limit N]
    python -m bench.e5 score     --model sam --generator default --set DDR [--output any|oracle_class] [--score ...] [--cap train|0.005|0.02]
    python -m bench.e5 prompted  --model sam|sam2|medsam --kind box|point --set IDRiD [--k 1,5,10,20,all] [--limit N]
    python -m bench.e5 control   --pred-dir <prob dir> --model <paper name> --train-set DDR --seed 0 --set IDRiD [--k ...]
    python -m bench.e5 sam3      --set DDR --split val ;  sam3-select ;  sam3-compose --set IDRiD
    python -m bench.e5 evaluate  --dir <scored dir> [--dir ...] | --all [--force]

Order on the server: prompts for the seven sets; amg + score on DDR val first (thresholds), then the test sets;
evaluate --all after each batch (DDR_val directories are evaluated first so the val_opt thresholds exist).
"""

import argparse

from bench.e5 import common


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench.e5")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts")
    p.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    p.add_argument("--split", default="test")

    sub.add_parser("areacap")

    a = sub.add_parser("amg")
    a.add_argument("--model", required=True, choices=["sam", "sam2"])
    a.add_argument("--generator", default="default")
    a.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    a.add_argument("--split", default="test")
    a.add_argument("--geometry", default="G1", choices=["G1", "G2"])
    a.add_argument("--limit", type=int, help="smoke: first N images, no done marker")

    s = sub.add_parser("score")
    s.add_argument("--model", required=True, choices=["sam", "sam2"])
    s.add_argument("--generator", default="default")
    s.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    s.add_argument("--split", default="test")
    s.add_argument("--geometry", default="G1", choices=["G1", "G2"])
    s.add_argument("--output", default="any", choices=["any", "oracle_class"])
    s.add_argument("--score", default="predicted_iou", choices=["predicted_iou", "stability_score"])
    s.add_argument("--cap", default=None, help="train | fraction of the field-of-view area, e.g. 0.005")

    b = sub.add_parser("prompted")
    b.add_argument("--model", required=True, choices=["sam", "sam2", "medsam"])
    b.add_argument("--kind", required=True, choices=["box", "point"])
    b.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    b.add_argument("--split", default="test")
    b.add_argument("--k", default="1,5,10,20,all")
    b.add_argument("--geometry", default="G1", choices=["G1", "G2"])
    b.add_argument("--variant", default="vit_h", help="SAM 1 backbone (e5_frozen: vit_h)")
    b.add_argument("--limit", type=int)

    c = sub.add_parser("control")
    c.add_argument("--pred-dir", required=True)
    c.add_argument("--model", required=True, help="paper model name, e.g. HRNet-W48")
    c.add_argument("--train-set", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    c.add_argument("--split", default="test")
    c.add_argument("--k", default="1,5,10,20,all")
    c.add_argument("--name", help="directory name under runs/E5/b_control (default: parent of --pred-dir)")

    t = sub.add_parser("sam3")
    t.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    t.add_argument("--split", default="test")
    t.add_argument("--geometry", default="G1", choices=["G1", "G2"])
    t.add_argument("--limit", type=int)
    sub.add_parser("sam3-select")
    tc = sub.add_parser("sam3-compose")
    tc.add_argument("--set", required=True, choices=list(common.PAPER_SET))
    tc.add_argument("--split", default="test")

    e = sub.add_parser("evaluate")
    e.add_argument("--dir", action="append", default=[])
    e.add_argument("--all", action="store_true")
    e.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "prompts":
        from bench.e5 import prompts

        prompts.build(args.set, args.split)
    elif args.cmd == "areacap":
        from bench.e5 import rules

        rules.training_area_cap()
    elif args.cmd == "amg":
        from bench.e5 import amg

        amg.run(args.model, args.generator, args.set, args.split, args.geometry, args.limit)
    elif args.cmd == "score":
        from bench.e5 import rules

        cap = None if args.cap in (None, "", "none") else ("train" if args.cap == "train" else float(args.cap))
        rules.run(args.model, args.generator, args.set, args.split, args.geometry, args.output, args.score, cap)
    elif args.cmd == "prompted":
        from bench.e5 import prompted

        prompted.run(args.model, args.kind, args.set, args.split, prompted.parse_k(args.k), args.geometry, args.variant, args.limit)
    elif args.cmd == "control":
        from bench.e5 import control, prompted

        control.run(args.pred_dir, args.model, args.train_set, args.seed, args.set, args.split, prompted.parse_k(args.k), args.name)
    elif args.cmd == "sam3":
        from bench.e5 import sam3_text

        sam3_text.run(args.set, args.split, args.geometry, args.limit)
    elif args.cmd == "sam3-select":
        from bench.e5 import sam3_text

        sam3_text.select()
    elif args.cmd == "sam3-compose":
        from bench.e5 import sam3_text

        sam3_text.compose(args.set, args.split)
    elif args.cmd == "evaluate":
        from bench.e5 import score

        if args.all:
            score.evaluate_all(force=args.force)
        for directory in args.dir:
            score.evaluate_dir(directory, force=args.force)


if __name__ == "__main__":
    main()
