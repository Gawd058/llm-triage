Workload: 240 requests, rate 2.0 req/s, seed 7, time_scale 0.1
50% easy / 30% medium / 20% hard; 20% high / 60% normal / 20% low priority;
Poisson arrivals with a 3x burst in the middle third. Latencies in simulated seconds.

| strategy | served | shed (429) | failed | p50 s | p95 s | p95 high-pri s | quality met | share small/medium/large |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| round_robin | 85.4% | 0.0% | 14.6% | 0.93 | 35.01 | 35.97 | 79.0% | 40%/38%/22% |
| random | 88.3% | 0.0% | 11.7% | 1.30 | 32.12 | 35.36 | 75.0% | 38%/40%/23% |
| least_loaded | 100.0% | 0.0% | 0.0% | 0.84 | 4.86 | 4.80 | 65.8% | 66%/23%/11% |
| strongest | 23.3% | 0.0% | 76.7% | 31.48 | 41.43 | 37.47 | 100.0% | 0%/0%/100% |
| triage | 96.7% | 3.3% | 0.0% | 0.35 | 11.87 | 12.34 | 79.7% | 68%/26%/6% |

quality met = share of served requests answered by a model at or above the request's true tier.
Classifier agreement with the hand labels: 90.0% (optimistic: the labels and the heuristic were written by the same person).
triage quality misses: 22 because the classifier underestimated the prompt, 25 because the right model was busy and the request was degraded.
