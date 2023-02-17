import concurrent.futures
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import geopandas as gpd
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.warp
import requests
import shapely.geometry
import xarray as xr
import zarr
from osgeo import gdal
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import json


class Region:
    def __init__(
        self,
        key: str,
        name: str,
        start_date: str,
        end_date: str,
        left: float,
        bottom: float,
        right: float,
        top: float,
        standardization: dict[str, float] | None = None,
        resolution: float = 10.0,
        crs_epsg: int = 32633,
    ):
        self.key = key
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.resolution = resolution
        self.standardization = standardization

        self.crs = rio.crs.CRS.from_epsg(crs_epsg)

    def __repr__(self):
        return f"Region (key: {self.key})"

    def transform(self):
        return rio.transform.from_origin(self.left, self.top, self.resolution, self.resolution)

    def height(self):
        return int(np.ceil((self.top - self.bottom) / self.resolution))

    def width(self):
        return int(np.ceil((self.right - self.left) / self.resolution))

    def xr_coords(self) -> list[tuple[str, np.ndarray]]:
        return [
            (
                "y",
                np.linspace(self.bottom + self.resolution / 2, self.top - self.resolution / 2, self.height())[
                    ::-1
                ],
            ),
            ("x", np.linspace(self.left + self.resolution / 2, self.right - self.resolution / 2, self.width())),
        ]

    def bbox(self):
        return [self.left, self.bottom, self.right, self.top]

    def bbox_wgs84(self, buffer: float = 500):
        bbox_gdf = gpd.GeoSeries([shapely.geometry.box(self.left, self.bottom, self.right, self.top)], crs=self.crs)

        return bbox_gdf.buffer(buffer).to_crs(4326).total_bounds


def load_regions(filepath: Path = Path("./points.json")) -> list[Region]:

    with open(filepath) as infile:
        regions_dict = json.load(infile)

    regions = []

    for key in regions_dict:
        regions.append(
            Region(
                key=key,
                **regions_dict[key],
            )
        )

    return regions


def download_region_data(region: Region, n_workers: int | None = 1) -> Path:

    output_dir = Path(f"output/{region.key}/").absolute()

    scenes_dir = output_dir.joinpath("scenes")


    merged_scenes_path = output_dir.joinpath("merged_scenes.zarr")

    if merged_scenes_path.is_dir():
        return merged_scenes_path

    out_chunks = {"x": 256, "y": 256, "time": 1}
    bbox_wgs84 = region.bbox_wgs84(buffer=500)
    height = region.height()
    width = region.width()
    transform = region.transform()
    spatial_coords = region.xr_coords()

    import ee

    ee.Initialize()
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee.Geometry.BBox(*bbox_wgs84))
        .filterDate(region.start_date, region.end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    img_info_list = []
    for img in collection.getInfo()["features"]:
        img_info_list.append(img["properties"] | {k: v for k, v in img.items() if k != "properties"})

    image_infos = pd.DataFrame.from_records(img_info_list)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    print(f"Got info for {region}")

    temp_paths = []
    images = []
    for _, image_info in image_infos.iterrows():
        temp_path = scenes_dir.joinpath(f"{image_info['system:index']}.zarr")
        temp_paths.append(temp_path)
        if temp_path.is_dir():
            continue
        image = ee.Image(image_info["id"])

        images.append(
            {
                "image_info": image_info,
                "image": image,
                "temp_path": temp_path,
            }
        )

    url_lock = threading.Lock()

    def process(image_dict, progress_bar):
        image = image_dict["image"]
        image_info = image_dict["image_info"]
        temp_path = image_dict["temp_path"]
        pols = list(image_info["transmitterReceiverPolarisation"])
        # Useless warnings are triggered by GDAL because of bad geotiffs by EE
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        with url_lock:
            url = image.getDownloadURL(
                {
                    "bands": pols,
                    "region": ee.Geometry.BBox(*bbox_wgs84),
                    "scale": region.resolution,
                    "crs": "epsg:32633",
                    "format": "GEO_TIFF",
                }
            )

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: {str(response.content)[:200]}")

        with rio.io.MemoryFile(response.content) as memfile:
            with memfile.open() as raster:
                output = np.empty((len(pols), height, width), dtype="float32")
                rio.warp.reproject(
                    raster.read(),
                    destination=output,
                    src_crs=raster.crs,
                    dst_crs=region.crs,
                    src_transform=raster.transform,
                    dst_transform=transform,
                    resampling=rio.warp.Resampling.cubic_spline,
                )
                #plt.imshow(output[0, :, :], extent=[region.left, region.right, region.bottom, region.top])
                #plt.show()
                #raise NotImplementedError()

        time_coord = [("time", [image_info["system:time_start"]])]

        arrays = {
            "image_metadata": xr.DataArray(data=[image_info.to_json()], coords=time_coord),
            "relative_orbit_nr": xr.DataArray(data=[image_info["relativeOrbitNumber_start"]], coords=time_coord),
        }
        encoding = {}
        for i, pol in enumerate(pols):
            encoding[f"{image_info['orbitProperties_pass']}_{pol}"] = {"compressor": compressor}
            arrays[f"{image_info['orbitProperties_pass']}_{pol}"] = xr.DataArray(
                data=output[[i], :, :],
                coords=time_coord + spatial_coords,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_name = Path(temp_dir).joinpath("arr.zarr")
            xr.Dataset(arrays).chunk(out_chunks).to_zarr(temp_name, encoding=encoding)
            shutil.move(temp_name, temp_path)

        progress_bar.update()

    os.makedirs(scenes_dir, exist_ok=True)

    with tqdm(total=len(images), smoothing=0.1, desc="Downloading scenes") as progress_bar:
        if n_workers != 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda i: process(i, progress_bar=progress_bar), images))
        else:
            for image_dict in images:
                process(image_dict, progress_bar)

    data = xr.open_mfdataset(temp_paths, engine="zarr")

    for variable in data.data_vars:
        if any(s in variable for s in ["ASCENDING", "DESCENDING"]):
            data[variable].attrs.update(
                {
                    "GDAL_AREA_OR_POINT": "Area",
                    "_CRS": {"wkt": str(region.crs.to_wkt())},
                }
            )
            data[variable].attrs["times"] = data[variable].dropna("time", how="all")["time"].values


    data["image_metadata"] = data["image_metadata"].astype(str)

    data.attrs.update({
        "key": region.key,
        "name": region.name,
        "start_date": region.start_date,
        "end_date": region.end_date,
    })

    if merged_scenes_path.is_dir():
        shutil.rmtree(merged_scenes_path)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_path = Path(temp_dir_str).joinpath("arr.zarr")
        task = data.chunk(out_chunks).to_zarr(
            temp_path, mode="w", encoding={v: {"compressor": compressor} for v in data.variables}, compute=False
        )
        with TqdmCallback(desc="Writing file", smoothing=0.05):
            task.compute()

        shutil.move(temp_path, merged_scenes_path)

    return merged_scenes_path


def download_data(n_workers: int | None = 1):
    bounds = {
        "left": 544000,
        "bottom": 8638500,
        "right": 555000,
        "top": 8650000,
    }
    resolution = 10
    output_path = Path("data.zarr")

    transform = rio.transform.from_origin(bounds["left"], bounds["top"], resolution, resolution)
    out_chunks = {"x": 256, "y": 256, "time": 1}

    width = int(np.ceil((bounds["right"] - bounds["left"]) / resolution))
    height = int(np.ceil((bounds["top"] - bounds["bottom"]) / resolution))
    spatial_coords = [
        ("y", np.linspace(bounds["bottom"] + resolution / 2, bounds["top"] - resolution / 2, height)[::-1]),
        ("x", np.linspace(bounds["left"] + resolution / 2, bounds["right"] - resolution / 2, width)),
    ]
    crs = rio.crs.CRS.from_epsg(32633)

    bbox_gdf = gpd.GeoSeries([shapely.geometry.box(*bounds.values())], crs=crs)

    bbox_buffered_wgs84 = bbox_gdf.buffer(500).to_crs(4326).total_bounds

    import ee

    ee.Initialize()
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(ee.Geometry.BBox(*bbox_buffered_wgs84))
        .filterDate("2018-01-01", "2023-04-01")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    img_info_list = []
    for img in collection.getInfo()["features"]:
        img_info_list.append(img["properties"] | {k: v for k, v in img.items() if k != "properties"})
    image_infos = pd.DataFrame.from_records(img_info_list)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    # image_infos = image_infos.iloc[:1]

    print("Got info")

    temp_paths = []
    images = []
    for _, image_info in image_infos.iterrows():
        temp_path = Path(f"temp/{image_info['system:index']}.zarr")
        temp_paths.append(temp_path)
        if temp_path.is_dir():
            continue
        image = ee.Image(image_info["id"])

        images.append(
            {
                "image_info": image_info,
                "image": image,
                "temp_path": temp_path,
            }
        )

    url_lock = threading.Lock()

    def process(image_dict, progress_bar):
        image = image_dict["image"]
        image_info = image_dict["image_info"]
        temp_path = image_dict["temp_path"]
        pols = list(image_info["transmitterReceiverPolarisation"])
        # Useless warnings are triggered by GDAL because of bad geotiffs by EE
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        with url_lock:
            url = image.getDownloadURL(
                {
                    # "crs_transform": [resolution, 0, bounds["left"], 0, -resolution, bounds["top"]],
                    "bands": pols,
                    "region": ee.Geometry.BBox(*bbox_buffered_wgs84),
                    "scale": resolution,
                    "crs": "epsg:32633",
                    "format": "GEO_TIFF",
                }
            )

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"{response.status_code}: {str(response.content)[:200]}")

        with rio.io.MemoryFile(response.content) as memfile:
            with memfile.open() as raster:
                output = np.empty((len(pols), height, width), dtype="float32")
                rio.warp.reproject(
                    raster.read(),
                    destination=output,
                    src_crs=raster.crs,
                    dst_crs=crs,
                    src_transform=raster.transform,
                    dst_transform=transform,
                    resampling=rio.warp.Resampling.cubic_spline,
                )


        time_coord = [("time", [image_info["system:time_start"]])]

        arrays = {
            "image_metadata": xr.DataArray(data=[image_info.to_json()], coords=time_coord),
            "relative_orbit_nr": xr.DataArray(data=[image_info["relativeOrbitNumber_start"]], coords=time_coord),
        }
        encoding = {}
        for i, pol in enumerate(pols):
            encoding[f"{image_info['orbitProperties_pass']}_{pol}"] = {"compressor": compressor}
            arrays[f"{image_info['orbitProperties_pass']}_{pol}"] = xr.DataArray(
                data=output[[i], :, :],
                coords=time_coord + spatial_coords,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_name = Path(temp_dir).joinpath("arr.zarr")
            xr.Dataset(arrays).chunk(out_chunks).to_zarr(temp_name, encoding=encoding)
            shutil.move(temp_name, temp_path)

        progress_bar.update()

    with tqdm(total=len(images), smoothing=0.1) as progress_bar:
        if n_workers != 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda i: process(i, progress_bar=progress_bar), images))
        else:
            for image_dict in images:
                process(image_dict, progress_bar)

    data = xr.open_mfdataset(temp_paths, engine="zarr")

    for variable in data.data_vars:
        if any(s in variable for s in ["ASCENDING", "DESCENDING"]):
            data[variable].attrs.update(
                {
                    "GDAL_AREA_OR_POINT": "Area",
                    "_CRS": {"wkt": str(crs.to_wkt())},
                }
            )
            data[variable].attrs["times"] = data[variable].dropna("time", how="all")["time"].values

    if output_path.is_dir():
        shutil.rmtree(output_path)

    task = data.chunk(out_chunks).to_zarr(
        output_path, mode="w", encoding={v: {"compressor": compressor} for v in data.variables}, compute=False
    )
    with TqdmCallback(desc="Writing file", smoothing=0.05):
        task.compute()
    return

    datasets = []
    for _, image_info in tqdm(image_infos.iterrows(), total=image_infos.shape[0], smoothing=0.1):

        temp_path = Path(f"temp/{image_info['system:index']}.zarr")
        if temp_path.is_dir():
            continue
        image = ee.Image(image_info["id"])

    raise NotImplementedError()

    data = xr.combine_by_coords(datasets)
    for variable in data.data_vars:
        if any(s in variable for s in ["ASCENDING", "DESCENDING"]):
            data[variable].attrs.update(
                {
                    "GDAL_AREA_OR_POINT": "Area",
                    "_CRS": {"wkt": str(crs.to_wkt())},
                }
            )
            data[variable].attrs["times"] = data[variable].dropna("time", how="all")["time"].values

    if output_path.is_dir():
        shutil.rmtree(output_path)

    task = data.chunk(out_chunks).to_zarr(
        output_path, mode="w", encoding={v: {"compressor": compressor} for v in data.variables}, compute=False
    )
    with TqdmCallback(desc="Writing file", smoothing=0.05):
        task.compute()

    data = xr.open_zarr(output_path)


def plot_data(region: Region, band_name: str = "ASCENDING_HV", n_workers: int | None = 1):

    data_path = download_region_data(region=region)

    frame_dir = data_path.parent.joinpath("frames/")

    os.makedirs(frame_dir, exist_ok=True)
    data = xr.open_zarr(data_path)

    band = data[band_name]
    # Filter out missing values by the "times" attr
    band = band.sel(time=band.attrs["times"])

    # Convert times to datetime
    band["time"] = band["time"].astype("datetime64[ms]")
    band.attrs["times"] = np.array(band.attrs["times"]).astype("datetime64[ms]")
    band = band.resample(time="1D").mean().dropna("time", how="all").load()

    band = band.where(band != 0.0).interpolate_na("x")

    if region.standardization is not None:
        band /= -band.sel(
            x=slice(region.standardization["left"], region.standardization["right"]),
            y=slice(region.standardization["top"], region.standardization["bottom"]),
        ).mean(["x", "y"])

    read_lock = threading.Lock()

    def process(i, progress_bar) -> Path:

        with read_lock:
            image = band.isel(time=i)

        date = str(image.time.dt.date.values)
        out_path = frame_dir.joinpath(f"{band_name}/frame_{band_name}_{date}.jpg")

        fig = plt.figure(figsize=(8, 8.5))
        axis = fig.add_subplot(111)

        axis.imshow(
            image.values,
            cmap="Greys_r",
            vmin=-2,
            vmax=0.5,
            extent=[image.x.min(), image.x.max(), image.y.min(), image.y.max()],
            interpolation="bilinear",
        )
        # fig.subplots_adjust(left=0, right=1, bottom=0.03, top=0.98)
        fig.tight_layout()
        axis.text(0.5, 0.99, f"{region.name}: {date}", transform=fig.transFigure, fontsize=16, ha="center", va="top")

        fig.savefig(out_path, dpi=300)

        plt.close()

        progress_bar.update()

        return out_path


    frame_paths = []
    os.makedirs(frame_dir.joinpath(band_name), exist_ok=True)
    with tqdm(total=band.shape[0], smoothing=0.1, desc="Generating frames") as progress_bar:
        if n_workers != 1:
            raise NotImplementedError("Matplotlib warns about multiple processes failing")
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                frame_paths = list(executor.map(lambda i: process(i, progress_bar=progress_bar), range(band.shape[0])))

        else:
            for i in range(band.shape[0]):
                frame_paths.append(process(i, progress_bar=progress_bar))

    with tempfile.TemporaryDirectory() as temp_dir_str:

        for i, frame_path in enumerate(frame_paths):
            filename = Path(temp_dir_str).joinpath(f"frame_{str(i).zfill(4)}.jpg")
            shutil.copy(frame_path, filename)


        filename = f"{region.key}_{band_name.lower()}.mp4"
        print(f"Generating {filename}")
        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "30",
                "-y",
                "-pattern_type",
                "glob",
                "-i",
                f"{temp_dir_str}/*.jpg",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(data_path.parent.joinpath(filename)),
            ],
            stdout=subprocess.PIPE,
            check=True,
        )


def main():
    regions = load_regions()
    plot_data(regions[2], "ASCENDING_HH")

    return
    for region in regions:
        download_region_data(region, n_workers=None)

    for region in regions:
        plot_data(region)
    #plot_data(regions[-1])
    

if __name__ == "__main__":
    main()

