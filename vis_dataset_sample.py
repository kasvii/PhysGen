#!/usr/bin/env python3
"""
Visualize DrivAerNet++ samples (shape + ground-truth surface pressure) in the same format
as the SP-VAE test figures written by ``commands/test_totalvae_single_node.sh``.

Usage (from the PhysGen root):
    python vis_dataset_sample.py --uid N_S_WWS_WM_172
    python vis_dataset_sample.py --uid F_D_WM_WW_0001 E_S_WWC_WM_005 --out_dir outputs/vis_dataset

Only needs numpy, scipy, matplotlib and vtk (no GPU, no OpenGL).
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from finetune_all.visualization import DEFAULT_PRESSURE_RANGE, render_dataset_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uid", nargs="+", required=True, help="sample id(s), e.g. N_S_WWS_WM_172")
    parser.add_argument("--data_root", default="data/drivaernet_plus", help="DrivAerNet++ root (with pressure/PressureVTK)")
    parser.add_argument("--split_dir", default="data/dataset1", help="dir with train/val/test.json (to report the split)")
    parser.add_argument("--out_dir", default="outputs/vis_dataset", help="where to write <uid>.png")
    parser.add_argument("--pressure_range", type=float, nargs=2, default=DEFAULT_PRESSURE_RANGE,
                        metavar=("MIN", "MAX"), help="colour range of the pressure map [m^2/s^2]")
    parser.add_argument("--px_per_unit", type=float, default=260, help="image resolution (pixels per normalized unit)")
    args = parser.parse_args()

    failed = []
    for uid in args.uid:
        t = time.time()
        out_path = os.path.join(args.out_dir, f"{uid}.png")
        try:
            render_dataset_sample(uid, args.data_root, out_path, split_dir=args.split_dir,
                                  pressure_range=tuple(args.pressure_range), px_per_unit=args.px_per_unit)
            print(f"[{uid}] saved {out_path} ({time.time() - t:.1f}s)")
        except Exception as e:  # keep going for the remaining uids
            print(f"[{uid}] failed: {e}", file=sys.stderr)
            failed.append(uid)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
