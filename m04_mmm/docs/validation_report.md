# Validation Report: Marketing Mix Model

## Convergence Diagnostics

| Metric | Value | Target |
|--------|-------|--------|
| Max R-hat | 1.0067 | < 1.01 |
| Min ESS (bulk) | 2032 | > 400 |
| Min ESS (tail) | 1384 | > 400 |

## Out-of-Sample Performance

| Metric | Value |
|--------|-------|
| MAE | 9,915 EUR |
| MAPE | 3.9% |
| R-squared | 0.9145 |

## Parameter Recovery

### Adstock Alphas

| Channel | True | Estimated | HDI 3% | HDI 97% | In HDI |
|---------|------|-----------|--------|---------|--------|
| tv | 0.70 | 0.2313 | 0.0000 | 0.6032 | No |
| ooh | 0.50 | 0.2836 | 0.0001 | 0.7153 | Yes |
| print | 0.30 | 0.2847 | 0.0002 | 0.6984 | Yes |
| facebook | 0.20 | 0.2681 | 0.0000 | 0.6550 | Yes |
| search | 0.10 | 0.2744 | 0.0006 | 0.6324 | Yes |

### Saturation Lambdas

| Channel | True | Estimated | HDI 3% | HDI 97% | In HDI |
|---------|------|-----------|--------|---------|--------|
| tv | 0.50 | 2.1717 | 0.0425 | 5.2121 | Yes |
| ooh | 0.80 | 2.2275 | 0.0489 | 5.2772 | Yes |
| print | 1.00 | 2.1362 | 0.0204 | 5.1355 | Yes |
| facebook | 0.30 | 2.2863 | 0.0194 | 5.5484 | Yes |
| search | 0.40 | 2.5903 | 0.0567 | 6.0685 | Yes |
