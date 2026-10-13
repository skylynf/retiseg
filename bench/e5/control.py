"""Same-information control for tier b (docs/paper/16 E5b): a supervised model's maps restricted to the oracle boxes.

    python -m bench.e5 control --pred-dir runs/E3a/<run>/IDRiD/prob_test --model HRNet-W48 --train-set DDR --seed 0 \\
                               --set IDRiD [--split test] [--k 1,5,10,20,all] [--name <run>]

The oracle-prompt rows of the promptable models see the test labels through
the boxes. This control gives a trained model exactly the same information:
its probability map is kept inside the union of the first k oracle boxes of
each class and set to 0 elsewhere, then scored with the same protocol. The
gap between a promptable model with k boxes and a supervised model with the
same k boxes is then a comparison at equal label access. No model is run;
the maps come from the run's internal prob_test or its E3a directory.

Output: runs/E5/b_control/<name>/box/k<k>/<set>_<split>/{<CLS>/<id>.png, paper.json, done.json};
the results.csv key keeps the supervised model's name, training source and seed, variant oracle_control:box:<k>.
"""

import time
from pathlib import Path

import numpy as np

from bench.common.io import LESION_CLASSES, PROB_LEVELS, read_prob_quantized, write_prob
from bench.e5 import common, prompts
from bench.e5.prompted import K_DEFAULT


def out_dir(name, k, test_set, split):
    return common.OUT_ROOT / "b_control" / name / "box" / f"k{k}" / common.set_dirname(test_set, split)


def box_union(items, canvas_hw):
    region = np.zeros(canvas_hw, dtype=np.float32)
    for item in items:
        x0, y0, x1, y1 = item["box"]
        region[y0:y1, x0:x1] = 1.0
    return region


def run(pred_dir, model, train_set, seed, test_set, split="test", ks=K_DEFAULT, name=None):
    pred_dir = Path(pred_dir)
    name = name or pred_dir.parent.name
    table = prompts.load(test_set, split)
    ids = common.split_ids(test_set, split)
    dirs = {k: out_dir(name, k, test_set, split) for k in ks}
    pending = [k for k in ks if not common.is_done(dirs[k])]
    if not pending:
        return dirs
    started = time.perf_counter()
    for image_id in ids:
        canvas = common.load_canvas(test_set, image_id, masks=False)
        canvas_hw = tuple(canvas["canvas_hw"])
        for cls in LESION_CLASSES:
            prob = read_prob_quantized(pred_dir, cls, image_id).astype(np.float32) / PROB_LEVELS
            if prob.shape != tuple(canvas["original_hw"]):
                raise ValueError(f"{pred_dir}/{cls}/{image_id}.png is {prob.shape}, the image is {canvas['original_hw']}")
            items = table["images"][image_id][cls]
            for k in pending:
                region = common.place_on_original(box_union(prompts.select(items, k), canvas_hw), canvas) > 0
                write_prob(dirs[k], cls, image_id, np.where(region, prob, np.float32(0)))
    for k in pending:
        record = common.paper_record(model, f"oracle_control:box:{k}", test_set)
        record.update({"model": model, "train_set": train_set, "seed": int(seed)})
        common.write_json(dirs[k] / "paper.json", record)
        common.write_json(common.done_marker(dirs[k]), {
            "experiment": "E5", "tier": "b_control", "reads_labels": True, "source_pred_dir": str(pred_dir), "model": model,
            "train_set": train_set, "seed": int(seed), "k": k, "test_set": test_set, "split": split, "n_images": len(ids),
            "prompts_sha256": table.get("sha256"), "seconds": round(time.perf_counter() - started, 1),
        })
    print(f"{name} {test_set}_{split}: control maps for k {pending}", flush=True)
    return dirs
