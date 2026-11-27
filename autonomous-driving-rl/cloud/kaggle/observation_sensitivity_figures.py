"""Plot verified offline observation-sensitivity tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cloud.kaggle.common import read, sha, verify, write
from cloud.kaggle.observation_capture import OUTPUT

COLORS = {"stopped": "#c1433e", "low": "#946926", "moving": "#21678c"}


def curves(ax, rows, x, y, order=None):
    for label, color in COLORS.items():
        selected = rows[rows["speed_band"] == label]
        if selected.empty:
            continue
        grouped = selected.groupby(x)[y]
        median = grouped.median()
        if order is not None:
            median = median.reindex(order).dropna()
        indices = median.index
        q = grouped.quantile([.25, .75]).unstack().reindex(indices)
        plot_x = list(map(str, indices)) if order is not None else indices.to_numpy(dtype=float)
        ax.plot(plot_x, median.values, marker="o", ms=3, lw=1.6, color=color, label=label)
        ax.fill_between(plot_x, q[.25].values, q[.75].values, alpha=.11, color=color)
    ax.axhline(0, color="#687783", lw=.8)
    if ax.lines:
        ax.grid(axis="y", alpha=.18)


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.out
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    for filename, evidence in manifest["report_files"].items():
        verify(root / filename, evidence["sha256"])

    def table(name):
        return pd.read_csv(root / f"{name}.csv")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    trace = table("temporal_braking_trace").query("case == 'extreme_exact'")
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6), layout="constrained")
    fig.suptitle("Extreme replay | original actor and verified sensor returns")
    ax[0].plot(trace.t_s, trace.throttle_brake, color="#c1433e", lw=1)
    ax[0].axhline(0, color="gray", lw=.8)
    ax[0].set_ylabel("Throttle / brake")
    ax[1].plot(trace.t_s, trace.speed_kmh, color="#21678c", label="Ego speed")
    ax[1].set_ylabel("Ego speed (km/h)")
    ax[1].twinx().plot(trace.t_s, trace.target_x_m, color="#946926", alpha=.65, label="Target x")
    ax[1].figure.axes[-1].set_ylabel("Target longitudinal x (m)", color="#946926")
    ax[2].plot(trace.t_s, trace.confirmed_active_rays, color="#21678c", label="Active NPC")
    ax[2].plot(trace.t_s, trace.confirmed_pending_rays, color="#c1433e", label="Pending NPC")
    ax[2].set_ylabel("Verified first-hit rays")
    ax[2].legend(loc="upper right", frameon=False)
    ax[2].set_xlabel("Simulation time (s)")
    start = trace.loc[trace.landmark.fillna("").str.contains("last_stop_start"), "t_s"].iloc[0]
    for axis in ax:
        axis.axvspan(start, trace.t_s.max(), color="#adbccd", alpha=.23)
    save(fig, root / "temporal_braking.png")

    traffic_distances = table("traffic_distance_sensitivity")
    distances = traffic_distances.query(
        "case == 'extreme_exact' and variant == 'change_target_distance'")
    velocity = table("relative_velocity_sensitivity").query("case == 'extreme_exact'")
    lateral = table("observation_sensitivity").query(
        "case == 'extreme_exact' and variant == 'change_target_position'")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    fig.suptitle("Extreme offline input probes | median and interquartile range")
    curves(axs[0, 0], distances, "value", "counterfactual_throttle_brake")
    curves(axs[0, 1], distances, "value", "counterfactual_steering")
    axs[0, 0].set(xlabel="Target distance (m)", ylabel="Throttle / brake")
    axs[0, 1].set(xlabel="Target distance (m)", ylabel="Steering")
    axs[0, 0].legend(title="Ego speed band", frameon=False)
    relative = velocity[velocity.setting.isin(("stopped", "same_speed", "slower", "faster", "pulling_away"))].copy()
    relative["relative_mps_bin"] = np.clip(np.rint(relative.value).astype(int), -16, 8)
    curves(axs[1, 0], relative, "relative_mps_bin", "delta_throttle_brake")
    axs[1, 0].set(xlabel="Relative longitudinal velocity (m/s, binned)", ylabel="Δ throttle / brake")
    curves(axs[1, 1], lateral, "setting", "delta_steering",
           ["adjacent_right", "front", "adjacent_left", "behind", "oncoming"])
    axs[1, 1].set(ylabel="Δ steering", xlabel="Target placement at matched range")
    axs[1, 1].tick_params(axis="x", labelrotation=18)
    save(fig, root / "traffic_sensitivity.png")

    edge = traffic_distances.query(
        "case == 'extreme_exact' and variant == 'range_edge_fourth_actor' and step in (2456, 2973)")
    if edge.empty:
        raise ValueError("Missing the two exact-case range edge probes")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    fig.suptitle("The fourth TrafficContext actor at the 50 m inclusion threshold")
    for axis, step in zip(axs, (2456, 2973)):
        selected = edge[edge.step == step].sort_values("value")
        axis.plot(selected.value, selected.counterfactual_throttle_brake, color="#21678c", marker="o", ms=4)
        axis.axvline(50., color="#c1433e", lw=1, ls="--", label="50 m cutoff")
        axis.axhline(0, color="gray", lw=.8)
        axis.set(xlabel="Edited NPC range (m)", title=f"Decision input {step}")
    axs[0].set_ylabel("Deterministic throttle / brake")
    axs[1].legend(frameon=False)
    save(fig, root / "traffic_range_edge.png")

    boundary = table("boundary_sensitivity").query("variant == 'redistribute_road_offsets'")
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    fig.suptitle("Paired road-edge probes | original heading and other inputs retained")
    qualified = boundary[boundary.scene.isin(("near_edge_parallel", "near_edge_outward", "lane_center", "curve"))]
    for column, side in enumerate(("left", "right")):
        rows = qualified[qualified.setting == side]
        for row_index, y in enumerate(("delta_steering", "delta_throttle_brake")):
            axis = axs[row_index, column]
            for scene, subset in rows.groupby("scene"):
                grp = subset[subset.value <= 2.75].groupby("value")[y].median()
                axis.plot(grp.index, grp.values, marker="o", ms=3, label=scene.replace("_", " "))
            axis.axhline(0, color="#687783", lw=.8)
            axis.set_ylabel("Δ steering" if row_index == 0 else "Δ throttle / brake")
            axis.set_title(side.capitalize() + " road edge")
        axs[1, column].set_xlabel("Counterfactual edge distance (m)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    save(fig, root / "boundary_sensitivity.png")

    files = ("temporal_braking.png", "traffic_sensitivity.png", "traffic_range_edge.png", "boundary_sensitivity.png")
    manifest["figures"] = {name: sha(root / name) for name in files}
    write(manifest_path, manifest)


if __name__ == "__main__":
    main()
