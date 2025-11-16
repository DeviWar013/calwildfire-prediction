"""
3D discrepancy visualization between fire susceptibility and facility index.

- Uses binned tiles (build_bins) to aggregate to coarser grid.
- Combined 3D plot: bar height = facility, color = fire.
- Fire-only and facility-only 3D histograms.
- Static PNG with vertical colorbar (fire).
- Rotating GIF of the combined plot (no colorbar in frames).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import imageio.v2 as imageio

# -------------------------------------------------------------------
# Paths – adjust if your actual paths differ
# -------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]

DISCREPANCY_FP = BASE / "output" / "analysis" / "discrepancy" / "tile_discrepancy.parquet"
TILES_FP = BASE / "data" / "processed" / "tiled_ca" / "ca_tiles_1km.shp"

OUT_DIR = BASE / "output" / "visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STATIC = OUT_DIR / "facility_fire_3d.png"
OUT_FIRE = OUT_DIR / "fire_3d.png"
OUT_FACILITY = OUT_DIR / "facility_3d.png"
OUT_GIF = OUT_DIR / "facility_fire_rot.gif"
GRID = 0.2

# -------------------------------------------------------------------
# Your original WORKING binning function – unchanged
# Assumes a global GRID variable already exists in your script.
# -------------------------------------------------------------------
def build_bins(tiles_gdf, df):
    tiles_proj = tiles_gdf.to_crs(3310)
    cent_proj = tiles_proj.geometry.centroid
    cent = tiles_gdf.copy()
    cent["lon"] = cent_proj.to_crs(4326).x
    cent["lat"] = cent_proj.to_crs(4326).y
    cent = cent.drop(columns="geometry")

    m = df.merge(cent, on="tile_id", how="left").fillna(0)

    # uses global GRID defined elsewhere in your project
    m["lon_bin"] = (np.floor(m["lon"] / GRID) * GRID).astype("float32")
    m["lat_bin"] = (np.floor(m["lat"] / GRID) * GRID).astype("float32")

    agg = (
        m.groupby(["lon_bin", "lat_bin"])
         .agg(
            fire=("pred_fire_index", "mean"),
            facility=("facility_index", "mean"),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            count=("tile_id", "count")
         )
         .reset_index()
    )

    # Normalize both for visualization
    if agg["fire"].max() > 0:
        agg["fire"] = agg["fire"] / agg["fire"].max()
    if agg["facility"].max() > 0:
        agg["facility"] = agg["facility"] / agg["facility"].max()

    return agg


# -------------------------------------------------------------------
# Custom FIRE colormap (warm only, no purple/blue)
# -------------------------------------------------------------------
def build_fire_cmap():
    # You can tweak these to better match your handcrafted palette
    colors = [
        (0.35, 0.05, 0.00),   # dark red-brown (high)
        (0.90, 0.10, 0.05),   # strong red
        (0.98, 0.55, 0.15),   # orange
        (0.98, 0.85, 0.30),   # yellow
    ]
    cmap = LinearSegmentedColormap.from_list("fire_custom", colors, N=256)
    # If you found it was "reversed", flip it:
    cmap = cmap.reversed()
    return cmap


FIRE_CMAP = build_fire_cmap()


# -------------------------------------------------------------------
# Combined 3D plot: bars = facility height, color = fire
# -------------------------------------------------------------------
def plot_combined(ax, agg, fire_cmap, elev=30, azim=245):
    """
    Draw combined 3D barchart where:

    - bar position (x, y) = aggregated lon, lat
    - bar height = facility (normalized)
    - bar color = fire (normalized)
    """
    xs = agg["lon"].values
    ys = agg["lat"].values
    zs = np.zeros_like(xs)

    # Pre-normalized in build_bins; here we just take the columns
    facility = agg["facility"].values
    fire = agg["fire"].values

    # Bar size in lon/lat degrees – tuned for your GRID
    dx = dy = float(GRID) * 0.8

    colors_arr = fire_cmap(fire)

    ax.bar3d(
        xs, ys, zs,
        dx, dy, facility,
        color=colors_arr,
        shade=True,
        linewidth=0.1
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Normalized Facility Index")
    ax.set_title("California: Facility Capacity Shaded by Fire Susceptibility", fontsize=12)

    return fire  # return fire values for colorbar


# -------------------------------------------------------------------
# Separate 3D plots: fire-only / facility-only
# -------------------------------------------------------------------
def plot_separate(ax, agg, value_col, title, cmap, zlabel):
    xs = agg["lon"].values
    ys = agg["lat"].values
    zs = np.zeros_like(xs)

    vals = agg[value_col].values
    dx = dy = float(GRID) * 0.8

    ax.bar3d(
        xs, ys, zs,
        dx, dy, vals,
        color=cmap(vals),
        shade=True,
        linewidth=0.1
    )
    ax.view_init(elev=30, azim=245)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel(zlabel)
    ax.set_title(title)


# -------------------------------------------------------------------
# Static combined PNG with vertical colorbar (OPTION A)
# -------------------------------------------------------------------
def save_static(agg):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    fire_vals = plot_combined(ax, agg, FIRE_CMAP, elev=32, azim=242)

    # Create a ScalarMappable just for the colorbar
    norm = Normalize(vmin=0.0, vmax=1.0)
    mappable = cm.ScalarMappable(norm=norm, cmap=FIRE_CMAP)
    mappable.set_array(fire_vals)

    # Vertical colorbar on the right side
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.03,
        pad=0.08
    )
    cbar.set_label("Fire Susceptibility (normalized)", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT_STATIC, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# Separate PNGs for fire-only and facility-only
# -------------------------------------------------------------------
def save_separate_plots(agg):
    # Facility-only
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_separate(
        ax,
        agg,
        value_col="facility",
        title="California: Facility Index (3D Histogram)",
        cmap=colormaps["viridis"],
        zlabel="Normalized Facility Index"
    )
    # colorbar for facility
    norm_fac = Normalize(vmin=0.0, vmax=1.0)
    mappable_fac = cm.ScalarMappable(norm=norm_fac, cmap=colormaps["viridis"])
    mappable_fac.set_array(agg["facility"].values)
    cbar_fac = fig.colorbar(
        mappable_fac,
        ax=ax,
        fraction=0.03,
        pad=0.08
    )
    cbar_fac.set_label("Facility Index (normalized)", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT_FACILITY, dpi=300)
    plt.close(fig)

    # Fire-only
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_separate(
        ax,
        agg,
        value_col="fire",
        title="California: Fire Susceptibility (3D Histogram)",
        cmap=FIRE_CMAP,
        zlabel="Normalized Fire Susceptibility"
    )
    norm_fire = Normalize(vmin=0.0, vmax=1.0)
    mappable_fire = cm.ScalarMappable(norm=norm_fire, cmap=FIRE_CMAP)
    mappable_fire.set_array(agg["fire"].values)
    cbar_fire = fig.colorbar(
        mappable_fire,
        ax=ax,
        fraction=0.03,
        pad=0.08
    )
    cbar_fire.set_label("Fire Susceptibility (normalized)", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT_FIRE, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------------
# Rotating GIF – NO colorbar in frames
# -------------------------------------------------------------------
def save_rotation_gif(agg, elev=32, azim_start=0, azim_end=358, step=1, fps=14):
    tmp_dir = OUT_DIR / "tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for angle in range(azim_start, azim_end, step):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Same combined plot, no colorbar here
        plot_combined(ax, agg, FIRE_CMAP, elev=elev, azim=angle)

        plt.tight_layout()
        frame_path = tmp_dir / f"frame_{angle:03d}.png"
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)
        frame_paths.append(frame_path)

    # Build GIF
    frames = [imageio.imread(p) for p in frame_paths]
    # Use writer so we can specify loop=0 (infinite looping)
    with imageio.get_writer(OUT_GIF, mode="I", fps=fps, loop=0) as writer:
        for f in frames:
            writer.append_data(f)

    # Cleanup
    for p in frame_paths:
        p.unlink()
    tmp_dir.rmdir()


# -------------------------------------------------------------------
# main – small, clean, no huge plotting block
# -------------------------------------------------------------------
def main():
    print("Loading discrepancy data...")
    df = pd.read_parquet(DISCREPANCY_FP)

    print("Loading tiles...")
    tiles = gpd.read_file(TILES_FP)

    print("Building bins...")
    agg = build_bins(tiles, df)

    print("Saving static combined plot with colorbar...")
    save_static(agg)

    print("Saving separate fire/facility plots...")
    save_separate_plots(agg)

    print("Saving rotating GIF (no colorbar in frames)...")
    save_rotation_gif(agg)

    print("Done.")


if __name__ == "__main__":
    main()
