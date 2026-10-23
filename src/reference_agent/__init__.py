import sys

from reference_agent.graph import build_graph, default_model, describe_target

__all__ = ["build_graph", "default_model", "describe_target", "main"]


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: reference-agent TARGET DISTRACTOR1 DISTRACTOR2")
    print(describe_target(*sys.argv[1:]))
