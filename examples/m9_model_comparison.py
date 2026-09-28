from __future__ import annotations

from derivatives_engine.instruments.option import Option, OptionType
from derivatives_engine.research import (
    compare_european_pricers,
    run_asian_validation_study,
    run_barrier_validation_study,
    run_crr_convergence_study,
    run_early_exercise_study,
    run_monte_carlo_convergence_study,
    run_surface_validation_study,
)


def main() -> None:
    option = Option(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.01,
        option_type=OptionType.CALL,
    )

    crr = compare_european_pricers(option, steps=(10, 50, 100))
    mc = run_monte_carlo_convergence_study(option, budgets=(1000, 5000), seeds=(1000, 1001, 1002))
    premium = run_early_exercise_study(option, steps=(20, 50))
    asian = run_asian_validation_study(option, n_paths=2000, seed=1000)
    barrier = run_barrier_validation_study(option, barrier=110.0, n_paths=2000, seed=1000)
    surface = run_surface_validation_study(option=option, strike=100.0, maturity=1.0)

    print({"crr": len(crr), "mc": len(mc), "american": len(premium), "asian": len(asian), "barrier": len(barrier), "surface": len(surface)})


if __name__ == "__main__":
    main()
