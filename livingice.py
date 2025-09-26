import json
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pyproj
import itertools
import numpy as np
import shapely
from PIL import Image
import os


def main(poly_points_per_edge: int = 5):
    with open("points.json") as infile:
        points_raw = json.load(infile)


    points = []
    # start_crs = pyproj.CRS.from_epsg(32633)
    # end_crs = pyproj.CRS.from_epsg(4326)

    # transformer = pyproj.Transformer.from_crs(start_crs, end_crs)
    out_dir = Path("for_livingice")
    thumbnail_dir = out_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True, parents=True)

    for key, values in points_raw.items():
        if not values["enable"]:
            continue

        edges = ["left", "top", "right", "bottom"]
        bottomtop = ("bottom", "top")
        leftright = ("left", "right")

        # Generate points along the box (since it'll be warped by the CRS change)
        x_coords = np.empty((0,))
        y_coords = np.empty((0,))
        for edge in edges:
            coord_const = np.repeat(values[edge], poly_points_per_edge)
            first, second = leftright if edge in bottomtop else bottomtop
            coord_var = np.linspace(values[first], values[second], poly_points_per_edge)

            if edge in ["right", "bottom"]:
                coord_var = coord_var[::-1]

            x_coords = np.append(x_coords, (coord_const if edge in leftright else coord_var)[:-1])
            y_coords = np.append(y_coords, (coord_const if edge in bottomtop else coord_var)[:-1])
        poly = shapely.geometry.Polygon(np.transpose([x_coords, y_coords]))

        mode = values.get("mode", "DESCENDING_VV")

        dirname = Path(f"output/{key}/frames/{mode}")
        frames = sorted(dirname.glob(f"*{mode}*.jpg"))

        if len(frames) == 0:
            raise ValueError(f"Could not find any frames in {dirname}")
        last_frame = frames[-1]
        image = Image.open(last_frame)
        image.thumbnail((512, 512))

        image.save(thumbnail_dir / f"thumbnail_{key}.jpg")

        # if values["end_date"].lower() in ["", "none"]:
        values["start_date"] = frames[0].stem.split("_")[-1]
        values["end_date"] = last_frame.stem.split("_")[-1]

        points += [
            {key2: values[key2] for key2 in ["name", "start_date", "end_date"]} | {
                "mode": mode,
                "key": key,
                "animation_name": f"{key}_{mode.lower()}_lr_rev.webm",
                "geometry": poly,
            }
        ]

    points = pd.DataFrame.from_records(points)
    points = gpd.GeoDataFrame(points, crs=32633).to_crs(4326)
    points.to_file(out_dir / "surge_animations.geojson")

    animation_dir = out_dir / "animations"
    animation_dir.mkdir(exist_ok=True)
    for _, point in points.iterrows():

        animation_path = Path(f"output/{point['key']}/animations/{point['animation_name']}")

        if not animation_path.is_file():
            print(f"{animation_path} does not exist. Skipping.")
        elif not (link_path := animation_dir / animation_path.name).is_symlink():
            os.symlink(animation_path.absolute(), link_path) 


if __name__ == "__main__":
    main()
