import sys

from reference_agent.graph import build_graph, default_model, describe_target

__all__ = ["build_graph", "default_model", "describe_target", "main"]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-listener"]
    if len(args) != 3:
        sys.exit("usage: reference-agent [--no-listener] TARGET DISTRACTOR1 DISTRACTOR2")
    print(describe_target(*args, use_listener="--no-listener" not in sys.argv))
