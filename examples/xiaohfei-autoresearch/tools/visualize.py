"""Visualize autoresearch experiment results.

Reads the JSONL log written by ``agent.py`` (one ``ExperimentRecord`` per line)
and renders a self-contained, interactive HTML dashboard: the hill-climb score
progression across iterations, per-iteration training curves (mean ± std across
runs), and a browsable table of the ideas tried.

The output is a single HTML file with no runtime dependencies (data is embedded,
charts are inline SVG drawn by vanilla JS), so it opens anywhere and can be
shared as-is. The page is produced from a Jinja2 template in ``templates/``.

Usage:
    python visualize.py --input autoresearch_results.jsonl
    python visualize.py --input results.jsonl --output report.html --open

To view the report inside VSCode, use ``--serve``: it hosts the page on
localhost (which VSCode auto-forwards over a remote connection) and prints a URL
to paste into the built-in Simple Browser (Ctrl/Cmd-Shift-P -> "Simple Browser").
VSCode's Simple Browser only loads http(s):// URLs, so a plain ``file://`` open
of the HTML will not render -- serving over localhost is the way in.

    python visualize.py --input results.jsonl --serve
"""

import argparse
import http.server
import json
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"

# Metric keys treated as an x-axis (progress) rather than a plottable value,
# in order of preference for the default x-axis.
_X_PREFS = ["timesteps_used", "global_step", "step", "elapsed_time_s"]


def load_records(path: Path) -> list[dict]:
    """Parse the JSONL log into a list of records, skipping blank lines."""
    records: list[dict] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON — {exc}") from exc
    if not records:
        raise SystemExit(f"{path}: no records found")
    return records


def build_payload(records: list[dict]) -> dict:
    """Reshape raw records into a compact structure the front-end consumes.

    The metric keys logged per run vary between iterations (e.g. an idea that
    adds automatic entropy tuning logs ``alpha``/``alpha_loss``; one that removes
    the value network does not), so the set of plottable metrics is discovered
    from the data rather than hard-coded.
    """
    iterations: list[dict] = []
    metric_keys: set[str] = set()
    baseline = None

    for rec in records:
        result = rec.get("result", {})
        score = result.get("score", {})
        run_metrics = result.get("run_metrics", []) or []

        # ``best_score`` is the running best *after* this iteration; for a
        # rejected iteration 0 the best is unchanged, so it recovers the
        # pre-loop baseline score.
        if baseline is None and rec.get("iteration") == 0 and not rec.get("accepted"):
            baseline = rec.get("best_score")

        runs = []
        for run in run_metrics:
            for point in run:
                metric_keys.update(
                    k for k, v in point.items() if isinstance(v, (int, float))
                )
            runs.append(run)

        iterations.append(
            {
                "iteration": rec.get("iteration"),
                "summary": rec.get("idea", {}).get("summary", ""),
                "description": rec.get("idea", {}).get("description", ""),
                "branch": result.get("branch", ""),
                "mean": score.get("mean"),
                "std": score.get("std"),
                "num_runs": score.get("num_runs"),
                "accepted": bool(rec.get("accepted")),
                "best_score": rec.get("best_score"),
                "runs": runs,
            }
        )

    x_keys = [k for k in _X_PREFS if k in metric_keys]
    default_x = x_keys[0] if x_keys else None

    # Plottable metrics = numeric keys that aren't x-axes, reward first.
    ordered_metrics = sorted(metric_keys - set(_X_PREFS))
    if "cumulative_reward" in ordered_metrics:
        ordered_metrics.remove("cumulative_reward")
        ordered_metrics.insert(0, "cumulative_reward")
    default_metric = ordered_metrics[0] if ordered_metrics else None

    return {
        "iterations": iterations,
        "metrics": ordered_metrics,
        "xKeys": x_keys,
        "defaultX": default_x,
        "defaultMetric": default_metric,
        "baseline": baseline,
    }


def render_html(payload: dict, title: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template(_TEMPLATE_NAME)
    # ``data_json`` is embedded inside a <script type="application/json"> block;
    # escape "<" so a "</script>" inside any string can't close the tag early.
    data_json = json.dumps(payload).replace("<", "\\u003c")
    return template.render(title=title, data_json=data_json)


def serve(out_path: Path, port: int) -> None:
    """Serve ``out_path``'s directory on localhost and block until Ctrl-C.

    Binds to 127.0.0.1 so VSCode's remote port-forwarding picks it up; the
    printed URL loads in the built-in Simple Browser (which refuses file://).
    ``port=0`` lets the OS pick a free port, avoiding "address already in use"
    on repeated runs. If the requested ``port`` is already taken, fall back to
    an OS-picked free port rather than crashing.
    """
    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=str(out_path.parent)
    )
    # allow_reuse_address clears the TIME_WAIT "address in use" after a recent
    # run on the same port; a live listener on that port still raises EADDRINUSE,
    # so retry once on port 0 (OS picks any free port) before giving up.
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        if port == 0:
            raise
        print(f"port {port} is in use ({exc.strerror}); picking a free port.")
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)

    with httpd:
        chosen = httpd.server_address[1]
        url = f"http://127.0.0.1:{chosen}/{out_path.name}"
        print(f"\nserving {out_path.name} at {url}")
        print(
            "  -> in VSCode: Ctrl/Cmd-Shift-P, run 'Simple Browser: Show', "
            "paste the URL above."
        )
        print("  (Ctrl-C to stop)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the autoresearch results JSONL file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output HTML path (default: alongside input, .html extension).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated report in the default browser.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the report on localhost (for viewing in VSCode's Simple "
        "Browser) and block until Ctrl-C.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve (default: 8000; use 0 to let the OS pick).",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else in_path.with_suffix(".html")
    )

    records = load_records(in_path)
    payload = build_payload(records)
    html = render_html(payload, title=in_path.stem)
    out_path.write_text(html)

    n_it = len(payload["iterations"])
    n_acc = sum(1 for it in payload["iterations"] if it["accepted"])
    print(f"parsed {n_it} iteration(s), {n_acc} accepted -> {out_path}")

    if args.open:
        webbrowser.open(out_path.as_uri())

    if args.serve:
        serve(out_path, args.port)


if __name__ == "__main__":
    main()
