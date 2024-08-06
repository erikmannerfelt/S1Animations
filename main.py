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
import rasterio.crs
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
    ) -> None:
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

    def __repr__(self) -> str:
        return f"Region (key: {self.key})"

    def transform(self) -> rio.Affine:
        return rio.transform.from_origin(self.left, self.top, self.resolution, self.resolution)

    def height(self) -> int:
        return int(np.ceil((self.top - self.bottom) / self.resolution))

    def width(self) -> int:
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

    def bbox(self) -> list[float]:
        return [self.left, self.bottom, self.right, self.top]

    def bbox_wgs84(self, buffer: float = 500) -> list[float]:
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


def build_vrts(data_path: Path) -> None:

    data = xr.open_zarr(data_path)

    all_times = np.array(data.time.values)
    for pol in data.data_vars:
        if not any(k in str(pol) for k in ["ASCENDING", "DESCENDING"]):
            continue

        crs = rasterio.crs.CRS.from_wkt(data[pol].attrs["_CRS"]["wkt"])

        to_build = []
        for time_s in data[pol].attrs["times"]:
            i = np.argwhere(all_times == time_s).ravel()[0] 
            time = pd.to_datetime(time_s * 1e6).isoformat().replace(":", "-")

            filename = str(pol) + "_" + time + ".vrt"
            filepath = data_path.parent.joinpath(f"vrts/{pol}/{filename}")
            if filepath.is_file():
                continue

            to_build.append((i, filepath))

            
        if len(to_build) > 0:
            for i, filepath in tqdm(to_build, desc=f"Generating {pol}"):
                os.makedirs(filepath.parent, exist_ok=True)

                subprocess.run([
                    "gdalbuildvrt",
                    "-a_srs",
                f"epsg:{crs.to_epsg()}",
                filepath.absolute(),
                f"ZARR:{data_path.absolute()}:/{pol}:{i}"            
                ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                

        # for time_s in tqdm(data[pol].attrs["times"], desc=f"Generating {pol} vrts"):
        #     i = np.argwhere(all_times == time_s).ravel()[0] 
        #     time = pd.to_datetime(time_s * 1e6).isoformat().replace(":", "-")

        #     filename = pol + "_" + time + ".vrt"
        #     filepath = data_path.parent.joinpath(f"vrts/{pol}/{filename}")
        #     if filepath.is_file():
        #         continue

        #     os.makedirs(filepath.parent, exist_ok=True)

        #     subprocess.run([
        #         "gdalbuildvrt",
        #         "-a_srs",
        #     f"epsg:{crs.to_epsg()}",
        #     filepath.absolute(),
        #     f"ZARR:{data_path.absolute()}:/{pol}:{i}"            
        #     ],
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         check=True
        #     )

def download_region_data(region: Region, n_workers: int | None = 1, preview: bool = False, redo: bool = False) -> Path:

    if preview and n_workers != 1:
        raise ValueError("Call needs to be synchronous (n_workers=1) for preview")

    output_dir = Path(f"output/{region.key}/").absolute()

    scenes_dir = output_dir.joinpath("scenes")

    merged_scenes_path = output_dir.joinpath("merged_scenes.zarr")

    if merged_scenes_path.is_dir():
        if redo:
            shutil.rmtree(merged_scenes_path)
        else:
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
        .filter(ee.Filter.eq("platform_number", "A"))
    )

    img_info_list = []
    for img in collection.getInfo()["features"]:
        img_info_list.append(img["properties"] | {k: v for k, v in img.items() if k != "properties"})

    image_infos = pd.DataFrame.from_records(img_info_list).drop_duplicates(subset="system:time_start")

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    print(f"Fetching data for {region}. W: {width} px, H: {height} px")

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

        response = requests.get(url, timeout=90)

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
                if preview:
                    plt.imshow(output[0, :, :], extent=[region.left, region.right, region.bottom, region.top], cmap="Greys_r")
                    plt.show()

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

    data = xr.open_mfdataset(temp_paths, engine="zarr", combine="nested", concat_dim="time")
    print("Loaded data")

    for variable in data.data_vars:
        if any(s in variable for s in ["ASCENDING", "DESCENDING"]):
            data[variable].attrs.update(
                {
                    "GDAL_AREA_OR_POINT": "Area",
                    "_CRS": {"wkt": str(region.crs.to_wkt())},
                }
            )
            data[variable].attrs["times"] = data[variable].dropna("time", how="all")["time"].values

    data["image_metadata"] = data["image_metadata"].astype("<U4000")

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



def plot_data(region: Region, band_name: str = "ASCENDING_HV", n_workers: int | None = 1, redo: bool = False) -> None:

    data_path = download_region_data(region=region)

    frame_dir = data_path.parent.joinpath("frames/")

    os.makedirs(frame_dir, exist_ok=True)
    data = xr.open_zarr(data_path)

    def get_frame_filenames(i: int, times_arr = None) -> Path:

        if times_arr is None:
            times_arr = pd.to_datetime(data[band_name].attrs["times"], unit="ms")
        date = str(pd.Timestamp(times_arr[i]).date())
        return frame_dir.joinpath(f"{band_name}/frame_{band_name}_{date}.jpg")

    anim_filename = data_path.parent.joinpath(f"animations/{region.key}_{band_name.lower()}.mp4")
    anim_lr_filename = anim_filename.with_stem(anim_filename.stem + "_lr")
    anims_missing = (not anim_filename.is_file()) or (not anim_lr_filename.is_file())

    if (not anims_missing) and not redo:
      return

    anim_filename.parent.mkdir(exist_ok=True, parents=True)

    band = data[band_name]
    # Filter out missing values by the "times" attr
    band = band.sel(time=band.attrs["times"])

    # Convert times to datetime
    band["time"] = band["time"].astype("datetime64[ms]")
    band.attrs["times"] = np.array(band.attrs["times"]).astype("datetime64[ms]")
    band = band.resample(time="1D").mean().dropna("time", how="all").load()

    # After filtering out empty files, list the frame paths again
    frame_paths = [get_frame_filenames(i=i, times_arr=band.time.values) for i in range(band.shape[0])]
    frames_missing = not all(path.is_file() for path in frame_paths)

    band = band.where(band != 0.0).interpolate_na("x")

    if region.standardization is not None:
        standardization = band.sel(
            x=slice(region.standardization["left"], region.standardization["right"]),
            y=slice(region.standardization["top"], region.standardization["bottom"]),
        ).mean(["x", "y"]).interpolate_na("time").fillna(-1)
        band /= -standardization

    read_lock = threading.Lock()

    def process(i: int, out_path: Path) -> Path | None:

        with read_lock:
            image = band.isel(time=i).load()

        date = str(image.time.dt.date.values)

        if not np.any(np.isfinite(image.values)):
            return None

        fig = plt.figure(figsize=(8, 8.5))
        axis = fig.add_subplot(111)

        axis.imshow(
            image.values,
            cmap="Greys_r",
            vmin=-2,
            vmax=0.5,
            extent=(image.x.min(), image.x.max(), image.y.min(), image.y.max()),
            interpolation="bilinear",
        )
        # fig.subplots_adjust(left=0, right=1, bottom=0.03, top=0.98)
        fig.tight_layout()
        axis.text(0.5, 0.99, f"{region.name}: {date}", transform=fig.transFigure, fontsize=16, ha="center", va="top")

        fig.savefig(out_path, dpi=300)

        del fig

        plt.close()

        return out_path

    new_frame_paths = []
    missing_frames = []
    for i, frame in enumerate(frame_paths):
        if frame.is_file():
            new_frame_paths.append((i, frame))
        else:
            missing_frames.append((i, frame))
    if len(missing_frames) > 0:
        os.makedirs(frame_dir.joinpath(band_name), exist_ok=True)
        for i, filepath in tqdm(missing_frames, smoothing=0.1, desc="Generating frames"):
            result = process(i, filepath)

            if result is None:
                continue
            new_frame_paths.append((i, result))

    new_frame_paths.sort(key=lambda ip: ip[0])
    frame_paths = new_frame_paths

    if len(new_frame_paths) == 0:
        raise ValueError("Got no frames to generate animation")

    if (frames_missing or anims_missing) or redo:

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            output_path_gif = anim_filename.with_name(anim_filename.stem + "_lr.gif")

            for i, frame_path in frame_paths:
                filename = Path(temp_dir_str).joinpath(f"frame_{str(i).zfill(4)}.jpg")
                shutil.copy(frame_path, filename)


            print(f"Generating {anim_filename}")
            
            temp_anim = temp_dir / "anim.mp4"
            subprocess.run([
                "/usr/bin/env",
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
                str(temp_anim),
            ], capture_output=True, check=True)

            shutil.move(temp_anim, anim_filename)
            print(f"Generating {anim_lr_filename}")

            temp_anim_lr = temp_dir / "anim_lr.mp4"
            subprocess.run(
                [
                    "/usr/bin/env",
                    "ffmpeg",
                    "-i",
                    str(data_path.parent.joinpath(anim_filename)),
                    "-y",
                    "-vf",
                    "scale=trunc(iw/4)*2:trunc(ih/4)*2",
                    "-c:v",
                    "libx265",
                    "-crf",
                    "28",
                    str(temp_anim_lr),
                ],
                capture_output=True,
                check=True,
            )
            shutil.move(temp_anim_lr, anim_lr_filename)

            print(f"Generating {output_path_gif}")
            temp_gif = temp_dir / "anim.gif"
            subprocess.run(
                [
                    "/usr/bin/env",
                    "ffmpeg",
                    "-i",
                    str(anim_filename),
                    "-pix_fmt",
                    "gray",
                    "-filter_complex",
                    ",".join(
                        [
                            "reverse[r];[0][r]concat=n=2:v=1:a=0",
                            "fps=10",
                            "scale=480:-1:flags=lanczos",
                            "split [a][b];[a]palettegen [p];[b][p] paletteuse",
                        ]
                    ),
                    str(temp_gif),
                ],
                capture_output=True,
                check=True,
            )
            temp_gif_opt = temp_gif.with_stem("anim_opt")

            subprocess.run(
                [
                    "/usr/bin/env",
                    "gifsicle",
                    "-O5",
                    "--lossy=100",
                    str(temp_gif),
                    "-o",
                    str(temp_gif_opt),
                ],
                capture_output=True,
                check=True,
            )

            shutil.move(temp_gif_opt, output_path_gif)


def animate(frame_dir: Path, output_path: Path):

    if output_path.suffix != ".mp4":
        raise ValueError(f"Output path must be .mp4. Given: {output_path}")

    output_path_lr = output_path.with_stem(output_path.stem + "_lr")
    output_path_gif = output_path.with_name(output_path.stem + "_lr.gif")
    output_path_rev = output_path.with_name(output_path.stem + "_rev.webm")

    if output_path.is_file() and output_path_lr.is_file():
        return output_path, output_path_lr

    output_path.parent.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        for i, frame_path in enumerate(sorted(list(frame_dir.glob("*.jpg")))):
            filename = Path(temp_dir_str).joinpath(f"frame_{str(i).zfill(4)}.jpg")
            shutil.copy(frame_path, filename)

        print(f"Generating {output_path}")
        
        temp_anim = temp_dir / "anim.mp4"
        subprocess.run([
            "/usr/bin/env",
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
            str(temp_anim),
        ], capture_output=True, check=True)

        shutil.move(temp_anim, output_path)
        print(f"Generating {output_path_lr}")

        temp_anim_lr = temp_dir / "anim_lr.mp4"
        subprocess.run(
            [
                "/usr/bin/env",
                "ffmpeg",
                "-i",
                str(output_path),
                "-y",
                "-vf",
                "scale=trunc(iw/4)*2:trunc(ih/4)*2",
                "-c:v",
                "libx265",
                "-crf",
                "28",
                str(temp_anim_lr),
            ],
            capture_output=True,
            check=True,
        )
        shutil.move(temp_anim_lr, output_path_lr)

        print(f"Generating {output_path_gif}")
        temp_gif = temp_dir / "anim.gif"
        subprocess.run(
            [
                "/usr/bin/env",
                "ffmpeg",
                "-i",
                str(output_path),
                "-pix_fmt",
                "gray",
                "-filter_complex",
                ",".join(
                    [
                        "reverse[r];[0][r]concat=n=2:v=1:a=0",
                        "fps=10",
                        "scale=480:-1:flags=lanczos",
                        "split [a][b];[a]palettegen [p];[b][p] paletteuse",
                    ]
                ),
                str(temp_gif),
            ],
            capture_output=True,
            check=True,
        )
        temp_gif_opt = temp_gif.with_stem("anim_opt")

        subprocess.run(
            [
                "/usr/bin/env",
                "gifsicle",
                "-O5",
                "--lossy=100",
                str(temp_gif),
                "-o",
                str(temp_gif_opt),
            ],
            capture_output=True,
            check=True,
        )

        shutil.move(temp_gif_opt, output_path_gif)

        
        skip_secs = {"liestol": 4, "natascha": 4, "scheele": 3, "bore": 3, "vallakra": 1}

        extra = []
        for key in skip_secs:
            if key in str(output_path):
                extra += ["-ss", str(skip_secs[key])]

        temp_rev = temp_dir / "anim_rev.webm"
        subprocess.run(
            [
                "/usr/bin/env",
                "ffmpeg",
                *extra,
                "-i",
                str(output_path_lr),
                "-b:v",
                "-y",
                "-filter_complex",
                "'reverse[r];[0][r]concat=n=2:v=1:a=0'",
                str(temp_rev),
            ],
            capture_output=True,
            check=True,
        )
        shutil.move(temp_rev, output_path_rev)
            
        


    return output_path, output_path_lr
   


def main():
    regions = load_regions()

    redo = False
    try:
        update_raw = os.getenv("AD_UPDATE")
        if update_raw is not None:
            if str(update_raw).isnumeric():
                redo = bool(int(update_raw))
            else:
                redo = str(update_raw).lower() == "true"
                
    except Exception as e:
        raise e
    for region in regions:
        print(region)

        if region.key not in ["arnesen", "vallakra", "scheele", "liestol", "bore", "natascha", "johansen", "petermann", "sonklar", "negri", "doktor", "etonfront"]:
            continue

        filepath = download_region_data(region, n_workers=None, redo=redo)

        if region.key in ["scheele", "natascha", "vallakra", "etonfront", "bore", "liestol", "doktor", "arnesen"]:
            build_vrts(filepath)

        if region.key == "iskuras":
            continue

        # if region.key == "negri":
        #     build_vrts(filepath)

        if region.key == "scheele" and False:
            with xr.open_zarr(filepath) as data:
                from test_autorift import add_geometry

                add_geometry(data, region.key)

                asc = data["DESCENDING_VV"].sel(time=data["DESCENDING_VV"].attrs["times"]).groupby(data["centerline"]).mean()

                with TqdmCallback():
                    asc.compute()

                asc = asc.where((~asc.isnull().all("time")).compute(), drop=True)

                asc.transpose("centerline", "time").plot(vmin=-20, vmax=20, cmap="RdBu")
                plt.show()
                print(asc)
                return



                
            

        if region.key in ["ganskij", "etonfront"]:
            plot_data(region, "ASCENDING_HH", redo=redo)
            
        plot_data(region, "DESCENDING_VV", redo=redo)
if __name__ == "__main__":
    main()

