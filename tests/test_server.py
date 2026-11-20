from conftest import EASY, HARD, chat, stack, three_tier_config

from triage.resources import HostSample, StaticMonitor


async def test_easy_request_routed_to_small_with_explanation_headers():
    async with stack() as (client, _, _):
        r = await client.post("/v1/chat/completions", json=chat(EASY))
    assert r.status_code == 200
    assert r.headers["x-triage-backend"] == "small"
    assert "meets need 1" in r.headers["x-triage-reason"]
    assert r.json()["choices"][0]["message"]["content"].startswith("[small]")
    assert r.json()["model"] == "m-small"  # model name rewritten for the backend


async def test_hard_request_routed_to_large():
    async with stack() as (client, router, _):
        r = await client.post("/v1/chat/completions", json=chat(HARD))
    assert r.headers["x-triage-backend"] == "large"
    assert router.states["large"].successes == 1


async def test_pinning_by_model_name():
    async with stack() as (client, _, _):
        r = await client.post("/v1/chat/completions", json={**chat(EASY), "model": "medium"})
    assert r.headers["x-triage-backend"] == "medium"


async def test_legacy_completions_endpoint():
    async with stack() as (client, _, _):
        r = await client.post("/v1/completions", json={"model": "auto", "prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"].startswith("[small]")


async def test_streaming_passthrough():
    async with stack() as (client, router, _):
        r = await client.post("/v1/chat/completions", json=chat(EASY, stream=True, max_tokens=5))
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-triage-backend"] == "small"
        assert r.text.rstrip().endswith("data: [DONE]")
        assert r.text.count("data:") == 7  # 5 tokens + finish chunk + [DONE]
        state = router.states["small"]
        assert (state.inflight, state.successes) == (0, 1)


async def test_fallback_on_backend_failure_then_circuit_opens():
    async with stack(mock_options={"large": {"fail_rate": 1.0}}) as (client, router, _):
        first = await client.post("/v1/chat/completions", json=chat(HARD))
        assert first.status_code == 200
        assert first.headers["x-triage-backend"] == "medium"
        assert first.headers["x-triage-fallbacks"] == "1"

        await client.post("/v1/chat/completions", json=chat(HARD))  # 2nd failure trips breaker
        assert router.states["large"].breaker.state == "open"

        health = (await client.get("/health")).json()
        assert health["backends"]["large"] == "open"

        third = await client.post("/v1/chat/completions", json=chat(HARD))
        assert third.headers["x-triage-backend"] == "medium"
        assert "x-triage-fallbacks" not in third.headers  # large is skipped up front now
        assert router.states["large"].failures == 2


async def test_all_backends_failing_returns_502():
    failing = {name: {"fail_rate": 1.0} for name in ("small", "medium", "large")}
    async with stack(mock_options=failing) as (client, _, _):
        r = await client.post("/v1/chat/completions", json=chat(EASY))
    assert r.status_code == 502
    assert len(r.json()["error"]["attempts"]) == 3


async def test_client_errors_pass_through_without_hurting_health():
    async with stack() as (client, router, _):
        r = await client.post("/v1/chat/completions", json={"model": "auto", "messages": "oops"})
    assert r.status_code == 400
    assert router.states["small"].failures == 0
    assert router.states["small"].breaker.state == "closed"


async def test_low_priority_shed_when_overloaded():
    cfg = three_tier_config(policy={"shed_low_at": 0.0})
    async with stack(cfg) as (client, router, mocks):
        r = await client.post(
            "/v1/chat/completions", json=chat(EASY), headers={"x-priority": "low"}
        )
        ok = await client.post(
            "/v1/chat/completions", json=chat(EASY), headers={"x-priority": "high"}
        )
    assert r.status_code == 429
    assert r.headers["retry-after"] == "1"
    assert r.json()["error"]["type"] == "triage_overloaded"
    assert ok.status_code == 200
    assert router.shed["low"] == 1
    assert mocks["small"].state.stats["served"] == 1


async def test_cpu_pressure_shifts_hard_work_to_lighter_model():
    hot = StaticMonitor(HostSample(cpu_percent=100, ram_available_mb=16000))
    async with stack(monitor=hot) as (client, _, _):
        r = await client.post("/v1/chat/completions", json=chat(HARD))
    assert r.headers["x-triage-backend"] != "large"
    assert "host pressure 1.00" in r.headers["x-triage-reason"]


async def test_explain_is_a_dry_run():
    async with stack() as (client, _, mocks):
        r = await client.post("/triage/explain", json=chat(HARD))
    body = r.json()
    assert body["decision"]["target"] == "large"
    assert body["profile"]["difficulty"] >= 0.55
    assert len(body["decision"]["candidates"]) == 3
    assert all(m.state.stats["served"] == 0 for m in mocks.values())


async def test_models_status_metrics_and_dashboard():
    async with stack() as (client, _, _):
        await client.post("/v1/chat/completions", json=chat(EASY))
        models = (await client.get("/v1/models")).json()
        status = (await client.get("/triage/status")).json()
        metrics = (await client.get("/metrics")).text
        dashboard = await client.get("/dashboard")
    assert [m["id"] for m in models["data"]] == ["auto", "small", "medium", "large"]
    assert status["recent"][0]["backend"] == "small"
    assert status["recent"][0]["outcome"] == "success"
    assert 'triage_requests_total{backend="small",outcome="success"} 1' in metrics
    assert "triage_backend_circuit_state" in metrics
    assert dashboard.status_code == 200
    assert "<title>llm-triage</title>" in dashboard.text


async def test_invalid_json_is_rejected():
    async with stack() as (client, _, _):
        r = await client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        arr = await client.post("/v1/chat/completions", json=[1, 2])
    assert r.status_code == 400
    assert arr.status_code == 400
