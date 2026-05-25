"""CLI entry point. Use the math helpers below to test IntelliSense (F12, hover, Ctrl+Space)."""

from __future__ import annotations

import argparse


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Return a times b."""
    return a * b


def square(x: float) -> float:
    """Return x squared — calls multiply."""
    return multiply(x, x)


def hypotenuse(a: float, b: float) -> float:
    """Length of hypotenuse for legs a, b — uses add and square."""
    return (add(square(a), square(b))) ** 0.5


def rectangle_area(width: float, height: float) -> float:
    """Area of a rectangle — calls multiply."""
    return multiply(width, height)


def demo_math() -> None:
    """Run a few chained calls so you can trace with Go to Definition."""
    w, h = 3.0, 4.0
    area = rectangle_area(w, h)
    diag = hypotenuse(w, h)
    total = add(area, diag)
    print(f"area={area}, diagonal={diag}, total={total}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gaze-mouse")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run math demo (IntelliSense test)")
    sub.add_parser("preview", help="Webcam + landmarks (Phase 1)")
    sub.add_parser("calibrate", help="Calibration UI (Phase 2)")
    sub.add_parser("train", help="Train model (Phase 3–4)")
    sub.add_parser("run", help="Live mouse control (Phase 5)")

    args = parser.parse_args(argv)

    if args.command == "demo":
        demo_math()
        return 0

    print(f"'{args.command}' not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
