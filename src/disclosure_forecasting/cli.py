import argparse
import json

from .facts import load_audit
from .train import execute, plan, read_config


def main():
    parser = argparse.ArgumentParser(
        description="Disclosure-process research; no automatic training"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="check version lineage and timestamp structure")
    audit.add_argument("csv")
    train = commands.add_parser("train", help="print a plan; --execute is required for training")
    train.add_argument("config")
    train.add_argument("--execute", action="store_true")
    train.add_argument("--output", default="runs/manual")
    train.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    train.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "bf16"))
    train.add_argument("--resume", action="store_true")
    train.add_argument("--eval-dir")
    build = commands.add_parser("build-eval", help="freeze the monitor/test evaluation worlds")
    build.add_argument("directory")
    build.add_argument("--monitor", type=int, default=256)
    build.add_argument("--test", type=int, default=4096)
    build.add_argument("--processes", type=int, default=8)
    build.add_argument("--families", default="stage-s", choices=("stage-s", "stage-s2"))
    trees = commands.add_parser("baselines", help="fit the boosted-tree references (CPU)")
    trees.add_argument("output")
    trees.add_argument("--eval-dir", required=True)
    trees.add_argument("--train-families", default="regular,varied")
    trees.add_argument("--worlds", type=int, default=200_000)
    trees.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    if args.command == "audit":
        _, report = load_audit(args.csv)
        print(json.dumps(report, indent=2))
    elif args.command == "build-eval":
        from .data import build_eval_sets
        from .simulator import HELDOUT_FAMILIES, S2_EVAL_FAMILIES, S2_NEW_HELDOUT

        sizes = {"monitor": args.monitor, "test": args.test}
        if args.families == "stage-s2":
            extra = {"families": S2_EVAL_FAMILIES, "heldout": HELDOUT_FAMILIES + S2_NEW_HELDOUT}
        else:
            extra = {}
        print(
            json.dumps(
                build_eval_sets(args.directory, sizes, processes=args.processes, **extra), indent=2
            )
        )
    elif args.command == "baselines":
        from .baselines import run

        run(args.output, args.eval_dir, args.train_families.split(","), args.worlds, args.processes)
    else:
        config = read_config(args.config)
        if args.execute:
            execute(config, args.output, args.device, args.precision, args.resume, args.eval_dir)
        else:
            print(json.dumps(plan(config), indent=2))


if __name__ == "__main__":
    main()
