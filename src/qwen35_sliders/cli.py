"""Bounded, explicit training and export commands. No automatic submission."""
import argparse
import json
from pathlib import Path

from .data import prepare_gsm


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-gsm")
    prepare.add_argument("source"); prepare.add_argument("output")
    prepare.add_argument("--train", type=int, default=128)
    prepare.add_argument("--dev", type=int, default=32)
    prepare.add_argument("--calibration", type=int, default=32)
    for name in ("train-particle", "train-lora", "search"):
        command = commands.add_parser(name)
        command.add_argument("config")
    export = commands.add_parser("build")
    export.add_argument("--workspace", required=True)
    export.add_argument("--deltas", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--strength", type=float, default=1.)
    export.add_argument("--norm-gain", type=float, default=1.)
    evaluate = commands.add_parser("evaluate-likelihood")
    evaluate.add_argument("--workspace", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--deltas")
    evaluate.add_argument("--strength", type=float, default=1.)
    evaluate.add_argument("--norm-gain", type=float, default=1.)
    evaluate.add_argument("--tasks", nargs="+")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--batch", type=int, default=4)
    math = commands.add_parser("evaluate-math")
    math.add_argument("--workspace", required=True)
    math.add_argument("--output", required=True)
    math.add_argument("--deltas")
    math.add_argument("--teacher")
    math.add_argument("--upstream")
    math.add_argument("--strength", type=float, default=1.)
    math.add_argument("--norm-gain", type=float, default=1.)
    math.add_argument("--limit", type=int, default=16)
    math.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "prepare-gsm":
        result = prepare_gsm(args.source, args.output, train=args.train, dev=args.dev, calibration=args.calibration)
        print(json.dumps({k:len(result[k]) for k in ("train", "dev", "calibration")}))
    elif args.command == "build":
        from .export import build
        print(json.dumps(build(args.workspace, args.deltas, args.output, strength=args.strength, norm_gain=args.norm_gain), indent=2))
    elif args.command == "evaluate-likelihood":
        from .evaluation import evaluate
        print(json.dumps(evaluate(args.workspace, args.output, deltas=args.deltas,
            strength=args.strength, norm_gain=args.norm_gain, names=args.tasks, limit=args.limit, batch=args.batch), indent=2))
    elif args.command == "evaluate-math":
        from .math_evaluation import evaluate_math
        print(json.dumps(evaluate_math(args.workspace, args.output, deltas=args.deltas,
            teacher=args.teacher, upstream=args.upstream, strength=args.strength, norm_gain=args.norm_gain,
            limit=args.limit, max_tokens=args.max_tokens), indent=2))
    elif args.command == "search":
        from .search import search
        search(json.loads(Path(args.config).read_text()))
    else:
        from .training import train_lora, train_particle
        function = train_particle if args.command == "train-particle" else train_lora
        print(json.dumps(function(json.loads(Path(args.config).read_text())), indent=2))


if __name__ == "__main__":
    main()
