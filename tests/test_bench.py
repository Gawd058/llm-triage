from triage.bench import STRATEGIES, format_report, make_workload, run_bench


def test_workload_is_reproducible():
    assert make_workload(50, 1.0, seed=3) == make_workload(50, 1.0, seed=3)
    assert make_workload(50, 1.0, seed=3) != make_workload(50, 1.0, seed=4)


async def test_bench_smoke():
    rows, jobs = await run_bench(n=24, rate=5.0, seed=1, time_scale=0.01, progress=False)
    assert [r["strategy"] for r in rows] == list(STRATEGIES)
    for row in rows:
        assert abs(row["served"] + row["shed"] + row["failed"] - 1.0) < 1e-9
    report = format_report(rows, jobs, "smoke")
    assert "| triage |" in report
