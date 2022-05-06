from typing import Tuple
from matplotlib import pyplot as plt
import pandas as pd
import os


def enrich_data(data: pd.DataFrame) -> pd.DataFrame:
    def get_verified_pct(tup: Tuple[int, int]) -> float:
        verified, correct = tup
        return (verified / correct) * 100 if correct > 0 else 0

    def get_activation(model_path: str) -> str:
        activations = ["ReLU", "ELU", "Sigmoid", "Tanh"]
        chunks = model_path.split("/")
        for activation in activations:
            if activation in chunks:
                return activation

    def get_adversarially_trained(model_path: str) -> bool:
        return os.path.basename(model_path).startswith("FFNN__Mix_")

    data["verified_pct"] = data[["verified", "correct"]].apply(get_verified_pct, axis=1)
    data["activation"] = data["model"].apply(get_activation)
    data["adversarially_trained"] = data["model"].apply(get_adversarially_trained)
    return data

def plot_vs_epsilon(data: pd.DataFrame, y_col: str, y_axis_label=None, title=None, ax=None, fig=None):
    plot_data = data.groupby(["activation", "adversarially_trained"]).agg({"epsilon": list, y_col: list})
    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        raise ValueError("Must supply a figure if you're supplying an axes object")
    colors = iter(["green", "darkgreen", "blue", "darkblue", "orange", "red"])
    for (activation, adversarially_trained), epsilon, ys in plot_data.itertuples():
        label = f"{activation}{', adv' if adversarially_trained else ''}"
        ax.plot(epsilon, ys, 'o-', label=label, color=next(colors))

    ax.set_xlabel("epsilon")
    ax.set_ylabel(y_axis_label if y_axis_label is not None else y_col)
    ax.set_title(title if title is not None else f"{ax.get_ylabel()} vs. epsilon")
    ax.legend()
    return fig, ax


if __name__ == "__main__":
    datafile = "experiments/experiment_2022-05-04T17:10:40.csv"
    data = pd.read_csv(datafile)
    data = enrich_data(data)

    fig1, _ = plot_vs_epsilon(data, "verified_pct", y_axis_label="% verified", title="")
    fig2, _ = plot_vs_epsilon(data, "verified", y_axis_label="# correct and verified")

    fig1.savefig("verified_pct_plot.png")
    fig2.savefig("num_verified_plot.png")
    plt.show()
