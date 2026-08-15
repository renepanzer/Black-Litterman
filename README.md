# Black-Litterman Portfolio Construction

A Black-Litterman allocation model built for a student-run investment fund. It combines an equilibrium prior with analyst return views and solves for optimal weights under the fund's investment mandate constraints.

## Method

Black-Litterman treats the market equilibrium as a prior and the analyst views as evidence, then produces a posterior expected return that blends the two:
$$
E[R] = [(\tau*\Sigma)^{-1} + P^{\mathsf{T}} \Omega^{-1} P]^{-1} [(\tau * \Sigma)^{-1} \pi + P^{\mathsf{T}} \Omega^{-1} Q]
$$

The code does the following:

1. Download four years of daily prices for the 25 asset universe using yfinance.
2. Estimate the Ledoit-Wolf shrunk covariance matrix and annualise it.
3. Reverse-optimise the weights to get the implied equilibrium excess returns `pi = risk_aversion * Sigma * w`.
4. Build the view matrix `P`, the view vector `Q` and the view uncertainty `Omega`. Views are absolute one-year return forecasts, converted to excess returns over a fixed risk-free rate. `Omega` is set proportional to the prior variance.
5. Combine prior and views into the posterior mean and covariance.
6. Maximise `mu'w - 0.5 * risk_aversion * w'Sigma w` subject to the mandate constraints, using OSQP through cvxpy.

## Constraints

| Constraint | Value |
|---|---|
| Long only | w >= 0 |
| Invested | 90%, with 10% held in cash |
| Single position | max 5% |
| Single sector | max 20% |

## Results

![Sector allocation](results/figures/Sector-Allocation.png)

![Asset allocation](results/figures/Asset-Allocation.png)

The optimal weights are written to `results/optimal_weights.xlsx`. Most positions receive the maximum 5% allocation as expected given the limited number of assets.
With a 90% budget and a 5% limit, at least 18 of the 25 assets must be held at the maximum, so the constraints do as much work as the expected returns in determining the final portfolio. That is why the asset chart is a bar chart rather than a pie: a pie would show a set of near-identical slices.

## Running it

```bash
pip install -r requirements.txt
python black_litterman.py
```

## Notes

The market weights are set to 1/N rather than market capitalisation. Reverse optimisation is meant to start from the market portfolio, so `pi` here is not a true equilibrium return.

Views are median sell-side analyst forecasts rather than the fund's own valuations, and `Omega` is set mechanically from the prior variance rather than from analyst conviction.

Prices are aligned with `dropna()`, so the sample starts at the first date on which every asset has a price. One asset in the universe listed partway through the window, which truncates the usable history.
