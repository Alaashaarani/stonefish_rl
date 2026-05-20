"""Plot the docking accuracy reward in x, y, z, reward space.

The docking environment currently uses:

    xy_error = norm([x_error, y_error])
    docking_accuracy = (xy_error ** 2) / z_abs_error

Because `xy_error ** 2 == x_error ** 2 + y_error ** 2`, this script plots:

    reward = -((x ** 2 + y ** 2) / z_abs_error)

The 4D plot is represented as a 3D scatter plot where x, y, and z are spatial
axes and reward is encoded by color.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_PATH = SCRIPT_DIR / "docking_accuracy_reward_4d.png"
DEFAULT_DIAGONAL_SAVE_PATH = SCRIPT_DIR / "docking_accuracy_diagonal_cut.png"
DEFAULT_HALF_SAVE_PATH = SCRIPT_DIR / "docking_accuracy_half_domain.png"
REWARD_MODES = ("accuracy", "xy-z")


def make_color_norm(values: np.ndarray, contrast_percentile: float = 2.0):
    """Return a color normalization that increases visible reward contrast.

    A few very low/high values can make most points look almost the same color.
    This uses percentile limits for the color scale while leaving the actual
    reward values unchanged.
    """
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None

    percentile = float(np.clip(contrast_percentile, 0.0, 49.0))
    if percentile > 0.0:
        vmin, vmax = np.percentile(finite_values, [percentile, 100.0 - percentile])
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))

    if np.isclose(vmin, vmax):
        return None

    return colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)


def docking_accuracy_reward(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    *,
    eps: float = 1e-3,
    clip_min: float | None = -25.0,
    normalize_z_like_env: bool = False,
) -> np.ndarray:
    """Return the reward corresponding to the docking accuracy term.

    Parameters
    ----------
    x_values, y_values, z_values:
        Error coordinates.
    eps:
        Small positive value used to avoid division by zero.
    clip_min:
        Lower clipping value for visualization. Use `None` to disable clipping.
    normalize_z_like_env:
        If true, mimic the environment's current `z_abs_error` preprocessing:
        `np.clip(z, 0.01, 5) / 5`. If false, use `abs(z)`.
    """
    xy_error_squared = x_values**2 + y_values**2

    if normalize_z_like_env:
        z_abs_error = np.clip(z_values, 0.01, 5.0) / 5.0
    else:
        z_abs_error = np.maximum(np.abs(z_values), eps)

    reward = -(xy_error_squared / z_abs_error)

    if clip_min is not None:
        reward = np.clip(reward, clip_min, 0.0)

    return reward


def xy_z_reward(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    *,
    eps: float = 1e-6,
    clip_min: float | None = None,
    normalize_z_like_env: bool = False,
) -> np.ndarray:
    """Return reward_xy + reward_z from the docking environment.

    The plotted equations are:

        reward_xy = clip(1 / (xy_error + 1e-6), 0, 1)
        reward_z = -clip(z_abs_error, 0, 1)
        reward = reward_xy + reward_z
    """
    xy_error = np.sqrt(x_values**2 + y_values**2)
    reward_xy = np.clip(1.0 / (xy_error + eps), 0.0, 1.0)

    if normalize_z_like_env:
        z_abs_error = np.clip(z_values, 0.01, 5.0) / 5.0
    else:
        z_abs_error = np.abs(z_values)

    reward_z = -np.clip(z_abs_error, 0.0, 1.0)
    reward = reward_xy + reward_z

    if clip_min is not None:
        reward = np.clip(reward, clip_min, 1.0)

    return reward


def compute_reward(
    reward_mode: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    *,
    clip_min: float | None,
    normalize_z_like_env: bool,
) -> np.ndarray:
    """Compute the selected reward field."""
    if reward_mode == "accuracy":
        return docking_accuracy_reward(
            x_values,
            y_values,
            z_values,
            clip_min=clip_min,
            normalize_z_like_env=normalize_z_like_env,
        )
    if reward_mode == "xy-z":
        return xy_z_reward(
            x_values,
            y_values,
            z_values,
            clip_min=clip_min,
            normalize_z_like_env=normalize_z_like_env,
        )
    raise ValueError(f"Unknown reward_mode '{reward_mode}'. Expected one of {REWARD_MODES}.")


def reward_title(reward_mode: str, normalize_z_like_env: bool) -> str:
    """Return a readable plot title for the selected reward."""
    if reward_mode == "xy-z":
        title = "Reward XY + Z: clip(1 / (xy_error + 1e-6), 0, 1) - clip(z_abs_error, 0, 1)"
    else:
        title = "Docking Accuracy Reward: reward = -((x^2 + y^2) / z_abs_error)"

    if normalize_z_like_env:
        title += "\nz_abs_error = clip(z, 0.01, 5) / 5"
    else:
        title += "\nz_abs_error = abs(z)"
    return title


def make_grid(x_limit: float, y_limit: float, z_min: float, z_max: float, points: int):
    """Create a regular 3D grid of x, y, z values."""
    x_axis = np.linspace(-x_limit, x_limit, points)
    y_axis = np.linspace(-y_limit, y_limit, points)
    z_axis = np.linspace(z_min, z_max, points)
    return np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")


def make_half_grid(x_limit: float, y_limit: float, z_min: float, z_max: float, points: int):
    """Create a half-domain grid with x from 0..x_limit and y from -y_limit..y_limit."""
    x_axis = np.linspace(0.0, x_limit, points)
    y_axis = np.linspace(-y_limit, y_limit, points)
    z_axis = np.linspace(z_min, z_max, points)
    return np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")


def plot_docking_reward_4d(
    *,
    x_limit: float = 5.0,
    y_limit: float = 5.0,
    z_min: float = 0.05,
    z_max: float = 5.0,
    points: int = 35,
    clip_min: float | None = -25.0,
    reward_mode: str = "accuracy",
    normalize_z_like_env: bool = False,
    alpha: float = 0.55,
    contrast_percentile: float = 2.0,
    save_path: str | Path = DEFAULT_SAVE_PATH,
    show: bool = True,
) -> Path:
    """Create and save a 4D reward plot.

    `points=35` creates 42,875 samples, which is a good balance between shape
    clarity and matplotlib speed.
    """
    x_grid, y_grid, z_grid = make_grid(x_limit, y_limit, z_min, z_max, points)
    reward = compute_reward(
        reward_mode,
        x_grid,
        y_grid,
        z_grid,
        clip_min=clip_min,
        normalize_z_like_env=normalize_z_like_env,
    )

    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = z_grid.ravel()
    reward_flat = reward.ravel()

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        x_flat,
        y_flat,
        z_flat,
        c=reward_flat,
        cmap="viridis_r",
        norm=make_color_norm(reward_flat, contrast_percentile),
        s=8,
        alpha=alpha,
        linewidths=0,
    )

    ax.set_title(reward_title(reward_mode, normalize_z_like_env))
    ax.set_xlabel("x error")
    ax.set_ylabel("y error")
    ax.set_zlabel("z error")
    ax.view_init(elev=24, azim=-45)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.68, pad=0.1)
    colorbar.set_label("reward")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_half_domain(
    *,
    x_limit: float = 5.0,
    y_limit: float = 5.0,
    z_min: float = 0.05,
    z_max: float = 5.0,
    points: int = 35,
    clip_min: float | None = -25.0,
    reward_mode: str = "accuracy",
    normalize_z_like_env: bool = False,
    alpha: float = 0.65,
    contrast_percentile: float = 2.0,
    save_path: str | Path = DEFAULT_HALF_SAVE_PATH,
    show: bool = True,
) -> Path:
    """Plot the half-domain reward field.

    This keeps the original 4D interpretation: x, y, and z are coordinates,
    while reward is the color. The sampled domain is:

        x in [0, x_limit]
        y in [-y_limit, y_limit]
        z in [z_min, z_max]
    """
    x_grid, y_grid, z_grid = make_half_grid(x_limit, y_limit, z_min, z_max, points)
    reward = compute_reward(
        reward_mode,
        x_grid,
        y_grid,
        z_grid,
        clip_min=clip_min,
        normalize_z_like_env=normalize_z_like_env,
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        x_grid.ravel(),
        y_grid.ravel(),
        z_grid.ravel(),
        c=reward.ravel(),
        cmap="viridis_r",
        norm=make_color_norm(reward.ravel(), contrast_percentile),
        s=10,
        alpha=alpha,
        linewidths=0,
    )

    ax.set_title("Half-Domain: x in [0, 5], y in [-5, 5]\n" + reward_title(reward_mode, normalize_z_like_env))
    ax.set_xlabel("x error")
    ax.set_ylabel("y error")
    ax.set_zlabel("z error")
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=24, azim=-58)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.68, pad=0.1)
    colorbar.set_label("reward")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_diagonal_cut(
    *,
    diagonal_limit: float = 5.0,
    z_min: float = 0.05,
    z_max: float = 5.0,
    points: int = 180,
    clip_min: float | None = -25.0,
    reward_mode: str = "accuracy",
    normalize_z_like_env: bool = False,
    contrast_percentile: float = 2.0,
    save_path: str | Path = DEFAULT_DIAGONAL_SAVE_PATH,
    show: bool = True,
) -> Path:
    """Plot a diagonal slice of the reward field where `x = y`.

    The horizontal axis is the diagonal displacement `d`, where:

        x = d
        y = d
        xy_error = sqrt(x^2 + y^2)

    Reward is shown as a colored surface over `(d, z)`.
    """
    diagonal_axis = np.linspace(-diagonal_limit, diagonal_limit, points)
    z_axis = np.linspace(z_min, z_max, points)
    diagonal_grid, z_grid = np.meshgrid(diagonal_axis, z_axis, indexing="ij")

    reward = compute_reward(
        reward_mode,
        diagonal_grid,
        diagonal_grid,
        z_grid,
        clip_min=clip_min,
        normalize_z_like_env=normalize_z_like_env,
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        diagonal_grid,
        z_grid,
        reward,
        cmap="viridis_r",
        norm=make_color_norm(reward, contrast_percentile),
        edgecolor="none",
        linewidth=0,
        antialiased=True,
    )

    ax.set_title("Diagonal Cut: x = y = d\n" + reward_title(reward_mode, normalize_z_like_env))
    ax.set_xlabel("diagonal displacement d, where x = y = d")
    ax.set_ylabel("z error")
    ax.set_zlabel("reward")
    ax.view_init(elev=28, azim=-130)

    colorbar = fig.colorbar(surface, ax=ax, shrink=0.68, pad=0.1)
    colorbar.set_label("reward")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot docking accuracy reward in 4D.")
    parser.add_argument("--x-limit", type=float, default=5.0)
    parser.add_argument("--y-limit", type=float, default=5.0)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--points", type=int, default=35)
    parser.add_argument(
        "--plot",
        choices=["half", "4d", "diagonal", "both"],
        default="half",
        help="Which plot to create. Default is the half-domain plot.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="accuracy",
        help="Reward equation to plot: 'accuracy' or 'xy-z'.",
    )
    parser.add_argument("--clip-min", type=float, default=-25.0)
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable reward clipping. This can make the color scale very wide.",
    )
    parser.add_argument(
        "--normalize-z-like-env",
        action="store_true",
        help="Use z_abs_error = clip(z, 0.01, 5) / 5, matching docking_env.py.",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--contrast-percentile",
        type=float,
        default=2.0,
        help=(
            "Percentile trim for color scaling. Higher values increase visible "
            "color differences but saturate more extreme rewards. Use 0 for raw scale."
        ),
    )
    parser.add_argument(
        "--save-path",
        default=str(DEFAULT_SAVE_PATH),
        help="Output image path for the 4D scatter plot.",
    )
    parser.add_argument(
        "--diagonal-save-path",
        default=str(DEFAULT_DIAGONAL_SAVE_PATH),
        help="Output image path for the diagonal cut plot.",
    )
    parser.add_argument(
        "--half-save-path",
        default=str(DEFAULT_HALF_SAVE_PATH),
        help="Output image path for the half-domain plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening the plot window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clip_min = None if args.no_clip else args.clip_min
    show = not args.no_show

    if args.plot in ("half", "both"):
        output = plot_half_domain(
            x_limit=args.x_limit,
            y_limit=args.y_limit,
            z_min=args.z_min,
            z_max=args.z_max,
            points=args.points,
            clip_min=clip_min,
            reward_mode=args.reward_mode,
            normalize_z_like_env=args.normalize_z_like_env,
            alpha=args.alpha,
            contrast_percentile=args.contrast_percentile,
            save_path=args.half_save_path,
            show=show,
        )
        print(f"Saved docking reward half-domain plot to: {output}")

    if args.plot in ("4d", "both"):
        output = plot_docking_reward_4d(
            x_limit=args.x_limit,
            y_limit=args.y_limit,
            z_min=args.z_min,
            z_max=args.z_max,
            points=args.points,
            clip_min=clip_min,
            reward_mode=args.reward_mode,
            normalize_z_like_env=args.normalize_z_like_env,
            alpha=args.alpha,
            contrast_percentile=args.contrast_percentile,
            save_path=args.save_path,
            show=show,
        )
        print(f"Saved docking reward 4D plot to: {output}")

    if args.plot in ("diagonal", "both"):
        diagonal_limit = min(args.x_limit, args.y_limit)
        output = plot_diagonal_cut(
            diagonal_limit=diagonal_limit,
            z_min=args.z_min,
            z_max=args.z_max,
            points=max(args.points, 80),
            clip_min=clip_min,
            reward_mode=args.reward_mode,
            normalize_z_like_env=args.normalize_z_like_env,
            contrast_percentile=args.contrast_percentile,
            save_path=args.diagonal_save_path,
            show=show,
        )
        print(f"Saved docking reward diagonal cut to: {output}")


if __name__ == "__main__":
    main()
