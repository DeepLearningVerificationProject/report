from matplotlib import pyplot
import math
from typing import Callable, Iterable
import numpy as np


def elu(x):
    """ELU activation function"""
    if x < 0:
        return math.exp(x) -1
    else:
        return x


def elu_deriv(x):
    """Derivative of ELU"""
    if x < 0:
        return math.exp(x)
    else:
        return 1


def elu_approx_1(lo, hi):
    """One alternative custom approximation for ELU between the lo and hi bounds"""
    upper_slope = (elu(hi) - elu(lo)) / (hi - lo)
    offset = hi * (1 - upper_slope)
    upper_bound = lambda x: x * upper_slope + offset
    lower_bound = lambda _: elu(lo)
    return lower_bound, upper_bound


def elu_approx_2(lo, hi):
    """Another alternative custom approximation for ELU between the lo and hi bounds"""
    _, upper_bound = elu_approx_1(lo, hi)
    lower_bound = lambda x: x
    return lower_bound, upper_bound


def general_elu_approx(lo, hi):
    """
    The general approximation used in the paper 'An abstract domain for certifying neural networks' for sigmoid and
    tanh, adapted for ELU. This approximation works for a function g(x) such that g(x) is twice differentiable,
    g'(x) > 0, and (0 <= g''(x)) --> (x <= 0) where --> is logical implication. The approximation is sound between
    the lo and hi bounds.
    """
    # lambda in the paper
    slope = (elu(hi) - elu(lo)) / (hi - lo)
    # lambda' in the paper
    min_deriv = min(elu_deriv(lo), elu_deriv(hi))

    if 0 <= lo:
        lower_bound = lambda x: elu(lo) + slope * (x - lo)
    else:
        lower_bound = lambda x: elu(lo) + min_deriv * (x - lo)

    if hi <= 0:
        upper_bound = lambda x: elu(hi) + slope * (x - hi)
    else:
        upper_bound = lambda x: elu(hi) + min_deriv * (x - hi)

    return lower_bound, upper_bound


def plot(fn: Callable[[float], float], xs: Iterable[float], ax=None):
    ys = [fn(x) for x in xs]
    if ax is None:
        return pyplot.plot(xs, ys)
    else:
        return ax.plot(xs, ys)


def approx_plots(x_min, x_max, x_bound_min, x_bound_max, points):
    """Plots all ELU approximations for comparison"""
    xs = np.linspace(x_min, x_max, points)

    low_bound_1, upper_bound_1 = elu_approx_1(x_bound_min, x_bound_max)
    low_bound_2, upper_bound_2 = elu_approx_2(x_bound_min, x_bound_max)
    low_bound_3, upper_bound_3 = general_elu_approx(x_bound_min, x_bound_max)
    bound_xs = np.linspace(x_bound_min, x_bound_max, points)

    axis_kwargs = {"color": "black", "linewidth": 0.5}
    bound_kwargs = {"linestyle": ":"}

    fig, (ax1, ax2, ax3) = pyplot.subplots(1, 3, sharex=True, sharey=True)
    plot(elu, xs, ax1)
    ax1.fill_between(bound_xs, [low_bound_1(x) for x in bound_xs], [upper_bound_1(x) for x in bound_xs], alpha=0.3)
    # axis lines
    ax1.axhline(**axis_kwargs)
    ax1.axvline(**axis_kwargs)
    # bound lines
    ax1.axvline(x=x_bound_min, **bound_kwargs)
    ax1.axvline(x=x_bound_max, **bound_kwargs)
    # bounding equations
    ax1.text(x_min,(elu(x_bound_max) + elu(x_bound_min)) / 2,"y = λ * x + μ")
    ax1.text((x_min + x_max) / 2 - 0.5, elu(x_bound_min) - 0.2, f"y = {low_bound_1(0):.2f}")

    ax1.set_title("Relaxation 1")

    plot(elu, xs, ax2)
    ax2.fill_between(bound_xs, [low_bound_2(x) for x in bound_xs], [upper_bound_2(x) for x in bound_xs], alpha=0.3)
    # axis lines
    ax2.axhline(**axis_kwargs)
    ax2.axvline(**axis_kwargs)
    # bound lines
    ax2.axvline(x=x_bound_min, **bound_kwargs)
    ax2.axvline(x=x_bound_max, **bound_kwargs)
    # bounding equations
    ax2.text(x_min, (elu(x_bound_max) + elu(x_bound_min)) / 2, "y = λ * x + μ")
    ax2.text((x_min + x_max) / 2 + 0.5, (x_bound_min + x_bound_max) / 2, "y = x")

    ax2.set_title("Relaxation 2")

    plot(elu, xs, ax3)
    ax3.fill_between(bound_xs, [low_bound_3(x) for x in bound_xs], [upper_bound_3(x) for x in bound_xs], alpha=0.3)
    # axis lines
    ax3.axhline(**axis_kwargs)
    ax3.axvline(**axis_kwargs)
    # bound lines
    ax3.axvline(x=x_bound_min, **bound_kwargs)
    ax3.axvline(x=x_bound_max, **bound_kwargs)

    ax3.set_title("General Relaxation")

    fig.suptitle(f"Alternative relaxations of ELU function between {x_bound_min} and {x_bound_max}")

    return ax1, ax2, ax3


def general_plots(x_min, x_max, x_bound_min, x_bound_max, points):
    """Only plots the general ELU approximation"""
    xs = np.linspace(x_min, x_max, points)

    low_bound, upper_bound = general_elu_approx(x_bound_min, x_bound_max)
    bound_xs = np.linspace(x_bound_min, x_bound_max, points)

    axis_kwargs = {"color": "black", "linewidth": 0.5}
    bound_kwargs = {"linestyle": ":"}

    fig, ax1 = pyplot.subplots(1, 1, sharex=True, sharey=True)
    plot(elu, xs, ax1)
    ax1.fill_between(bound_xs, [low_bound(x) for x in bound_xs], [upper_bound(x) for x in bound_xs], alpha=0.3)
    # axis lines
    ax1.axhline(**axis_kwargs)
    ax1.axvline(**axis_kwargs)
    # bound lines
    ax1.axvline(x=x_bound_min, **bound_kwargs)
    ax1.axvline(x=x_bound_max, **bound_kwargs)

    # bounding equations
    # lambda in the paper
    slope = (elu(x_bound_max) - elu(x_bound_min)) / (x_bound_max - x_bound_min)
    # lambda' in the paper
    min_deriv = min(elu_deriv(x_bound_min), elu_deriv(x_bound_max))

    if 0 < x_bound_min:
        # lower_bound_eq = f"elu({x_bound_min}) + λ * (x - {x_bound_min})"
        lower_bound_eq = f"{slope:.2f} * x + {elu(x_bound_min) - slope * x_bound_min:.2f}"
    else:
        # lower_bound_eq = f"elu({x_bound_min}) + λ' * (x - {x_bound_min})"
        lower_bound_eq = f"{min_deriv:.2f} * x + {elu(x_bound_min) - min_deriv * x_bound_min:.2f}"
    if x_bound_max <= 0:
        # upper_bound_eq = f"elu({x_bound_max}) + λ * (x - {x_bound_max})"
        upper_bound_eq = f"{slope:.2f} * x + {elu(x_bound_max) - slope * x_bound_max:.2f}"
    else:
        # upper_bound_eq = f"elu({x_bound_max}) + λ' * (x - {x_bound_max})"
        upper_bound_eq = f"{min_deriv:.2f} * x + {elu(x_bound_max) - min_deriv * x_bound_max:.2f}"

    mid_x = (elu(x_bound_max) + elu(x_bound_min)) / 2
    ax1.text(mid_x - 0.3, upper_bound(mid_x) + 0.2, upper_bound_eq, rotation=30)
    ax1.text(mid_x - 0.3, low_bound(mid_x) - 0.5, lower_bound_eq, rotation=30)

    ax1.set_title(f"ELU relaxation between {x_bound_min} and {x_bound_max}")

    return ax1


if __name__ == "__main__":
    points = 4000

    # approx_plots(-2, 2, -1, 1, points)
    general_plots(-2, 2, -1, 1, points)
    pyplot.savefig("elu_mixed_relaxation.png")
    pyplot.show()

    # approx_plots(-6, 2, -5, 1, points)
    # pyplot.savefig("elu_relaxation_2.png")
    # pyplot.show()

    # approx_plots(-2, 1, -1, 0, points)
    general_plots(-2, 1, -1, 0, points)
    pyplot.savefig("elu_exponential_relaxation.png")
    pyplot.show()

    # approx_plots(-1, 2, 0, 1, points)
    general_plots(-1, 2, 0, 1, points)
    pyplot.savefig("elu_linear_relaxation.png")
    pyplot.show()
