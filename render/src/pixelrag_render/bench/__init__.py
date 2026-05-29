"""pixelshot benchmark harness.

Usage:
    from pixelrag_render.strategies import CDPSequentialStrategy
    from pixelrag_render.bench import Bench

    bench = Bench(zim_path="...", chrome_path="...", output_dir="./results")
    result = await bench.run(CDPSequentialStrategy(chrome_path=..., n_workers=32, fmt="raw"))
"""

from .bench_throughput import (
    Bench as Bench,
    prepare_articles as prepare_articles,
    generate_ground_truth as generate_ground_truth,
    run_and_verify as run_and_verify,
)
