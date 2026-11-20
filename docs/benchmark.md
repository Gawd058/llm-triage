Workload: 240 requests, rate 0.8 req/s, seed 7, time_scale 0.1
50% easy / 30% medium / 20% hard; 20% high / 60% normal / 20% low priority;
Poisson arrivals with a 3x burst in the middle third. Latencies in simulated seconds.

| strategy | served | shed (429) | failed | p50 s | p95 s | p95 high-pri s | quality met | share small/medium/large |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| round_robin | 98.3% | 0.0% | 1.7% | 1.64 | 25.99 | 30.30 | 81.8% | 35%/33%/32% |
| random | 100.0% | 0.0% | 0.0% | 1.63 | 11.88 | 13.01 | 77.9% | 33%/35%/32% |
| least_loaded | 100.0% | 0.0% | 0.0% | 0.86 | 6.14 | 11.87 | 72.9% | 49%/36%/15% |
| strongest | 53.3% | 0.0% | 46.7% | 31.01 | 40.83 | 41.31 | 100.0% | 0%/0%/100% |
| triage | 100.0% | 0.0% | 0.0% | 0.35 | 11.92 | 11.93 | 82.9% | 68%/23%/9% |

quality met = share of served requests answered by a model at or above the request's true tier.
Classifier agreement with the hand labels: 90.0% (optimistic: the labels and the heuristic were written by the same person).
triage quality misses: 24 because the classifier underestimated the prompt, 17 because the right model was busy and the request was degraded.
