# Synthesis Methodology

## Campaign Event Generation
For each customer with >=5 transactions:
1. Draw number of sends from Poisson(lambda=2 per active month)
2. Draw send hour from truncated normal around modal purchase hour (+/- 3h)
3. Draw send day-of-week from customer historical day distribution
4. Compute open probability:
   - base open rate = 0.25
   - time alignment score = 1.0 within +/-2h, decays to 0.4
   - add Gaussian noise (sigma=0.05)
5. Draw `opened`, then conditional `clicked` (0.35), `purchased` (0.12)

## Reproducibility
- Fixed seed: 42
- Deterministic output sort before writing

## Notes
Synthetic data provides a controllable proxy signal for send-time optimization when real campaign telemetry is unavailable.
