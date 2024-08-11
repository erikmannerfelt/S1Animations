import geopandas as gpd
import glacier_lengths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(".cache")

def measure_lengths(glacier: str = "nataschabreen", use_cache: bool = True, radius: float = 200.):
    glacier_dir = Path(f"GIS/shapes/{glacier}")

    cache_filepath = CACHE_DIR / f"{glacier}_lengths.csv"

    if cache_filepath.is_file() and use_cache:
        lengths = pd.read_csv(cache_filepath)
    else:

        if (filepath := glacier_dir / "surge_front_positions.geojson").is_file():
            pass
        elif (filepath := glacier_dir / "front_positions.geojson").is_file():
            pass
        else:
            raise ValueError(f"No found front_positions.geojson in {glacier_dir}") 

        positions = gpd.read_file(filepath)

        centerline_shp = gpd.read_file(glacier_dir / "centerline.geojson")
        centerline = centerline_shp.iloc[0].geometry

        domain_shp = gpd.read_file(glacier_dir / "domain.geojson")
        domain = domain_shp.iloc[0].geometry

        buffered_centerlines = glacier_lengths.buffer_centerline(centerline, domain, max_radius=radius)

        lengths = []
        for (i, position) in positions.iterrows():

            for j, line in enumerate(buffered_centerlines.geoms):
                cut = glacier_lengths.cut_centerlines(line, position.geometry, max_difference_fraction=0.005)

                cut_lengths = glacier_lengths.measure_lengths(cut)

                lengths.append({
                    "line_i": j,
                    "date": position.date,
                    "length": cut_lengths[0],
                    # "median": np.median(cut_lengths),
                    # "std": np.std(cut_lengths),
                    # "upper": np.percentile(cut_lengths, 75),
                    # "lower": np.percentile(cut_lengths, 25),
                })

        lengths = pd.DataFrame.from_records(lengths)

        CACHE_DIR.mkdir(exist_ok=True)
        lengths.to_csv(cache_filepath, index=False)

    lengths["date"] = pd.to_datetime(lengths["date"])

    return lengths
    

def main(glacier: str = "vallakrabreen", use_cache: bool = True):

    centerline_radii = {
        "etonfront": 1200,
    }

    all_lengths = measure_lengths(glacier, use_cache=use_cache, radius=centerline_radii.get(glacier, 200)).sort_values("date").reset_index(drop=True)

    key_translation = {
        "vallakrabreen": "vallakra",
    }
    s1_key = key_translation.get(glacier, glacier)

    lengths = []
    for date, values in all_lengths.groupby("date"):

        lengths.append({
            "date": date,
            "median": np.median(values["length"]),
            "mean": np.mean(values["length"]),
            "std": np.std(values["length"]),
            "upper": np.percentile(values["length"], 75),
            "lower": np.percentile(values["length"], 25),
            "count": values.shape[0],
            

        })
    lengths = pd.DataFrame.from_records(lengths).sort_values("date")

    print(lengths)

    diff_intervals = pd.IntervalIndex.from_arrays(left=lengths["date"].iloc[:-1].values, right=lengths["date"].iloc[1:].values)


    diffs = []
    for i, interval in enumerate(diff_intervals):

        before_vals = all_lengths[all_lengths["date"] == interval.left].set_index("line_i")
        after_vals = all_lengths[all_lengths["date"] == interval.right].set_index("line_i")
         
        diff = (after_vals - before_vals).dropna()

        if diff.shape[0] == 0:
            continue

        diffs.append({
            "date": interval,
            "mean": np.mean(diff["length"]),
            "median": np.median(diff["length"]),
            "lower": np.percentile(diff["length"], 25),
            "upper": np.percentile(diff["length"], 75),
            "std": np.std(diff["length"]),
            "count": diff.shape[0],
        })


    diffs = pd.DataFrame.from_records(diffs).set_index("date").sort_index()

    diff_per_day = diffs / np.repeat(((diffs.index.right - diffs.index.left).total_seconds() / (3600 * 24)).values[:, None], diffs.shape[1], 1)
    diff_per_day["count"] = diffs["count"]

    diff_per_day["date_from"] = diff_per_day.index.left
    diff_per_day["date_to"] = diff_per_day.index.right

    diff_per_day = diff_per_day.set_index(diff_per_day.index.mid).resample("1M").mean().dropna()

    output_dir = Path(f"output/{s1_key}/")

    diff_per_day.to_csv(output_dir / f"{s1_key}_length_diff_per_day.csv")
    lengths.to_csv(output_dir / f"{s1_key}_length.csv")

    lengths[["lower", "upper", "mean", "median"]] -= lengths.iloc[0]["median"]

    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    plt.fill_between(lengths["date"], lengths["lower"], lengths["upper"], alpha=0.3)
    plt.plot(lengths["date"], lengths["median"])
    plt.ylabel("Length (m)")

    xlim = plt.gca().get_xlim()
    plt.hlines(0, *xlim, colors="black", linestyles=":", alpha=0.5, zorder=0)
    plt.xlim(xlim)
    plt.subplot(212)
    plt.fill_between(diff_per_day.index, diff_per_day["lower"], diff_per_day["upper"], alpha=0.3)
    plt.plot(diff_per_day.index, diff_per_day["median"])
    plt.ylabel("Advance speed (m/d)")
    xlim = plt.gca().get_xlim()
    plt.hlines(0, *xlim, colors="black", linestyles=":", alpha=0.5, zorder=0)
    plt.xlim(xlim)
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_dir / f"{s1_key}_length.jpg", dpi=600)



    

    
    plt.show()






if __name__ == "__main__":
    main()
