from autoRIFT import autoRIFT
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import pandas as pd
import warnings
import tempfile
import shutil
import zarr
import datetime
from osgeo import gdal, gdal_array
import rasterio.warp
import rasterio.features
import geopandas as gpd
from main import animate, build_vrts, download_region_data, load_regions, Region
import scipy.spatial
import shapely


def plot_velocities(velocity_data_path: Path):
    with xr.open_dataset(velocity_data_path, engine="zarr") as data:
        data["vy"] = data["vy"] - data["vy"].median(["x", "y"])
        data["vx"] = data["vx"] - data["vx"].median(["x", "y"])
        data["v"] = (data["vx"] ** 2 + data["vy"] ** 2) ** 0.5

        # Scheelebreen
        # point = data.sel(x=547940, y=8631397, method="nearest")

        # plt.show()
        # # Vallakrabreen
        # point = data.sel(x=549400, y=8642500, method="nearest")

        # point["v"].plot()
        # plt.show()

        # print(np.isfinite(data["v"].sel(time=slice(pd.Timestamp("2022-01-01"), None))).sum("time").plot())

        # plt.show()

        # return

        for time, arr in data.groupby("time"):
            plt.title(time)
            plt.imshow(arr["v"].isel(time=0), vmin=0, vmax=10, cmap="Reds")

            plt.show()

    # plt.figure(figsize=(16, 6))
    # plt.subplot(1, 6, 2)
    # plt.title("Velocity")
    # plt.imshow(vel, vmin=0, vmax=10, cmap="Reds")
    # plt.subplot(1, 6, 3)
    # plt.title("X velocity")
    # plt.imshow(x_vel, vmin=-10, vmax=10, cmap="RdBu")
    # plt.subplot(1, 6, 4)
    # plt.title("Y velocity")
    # plt.imshow(y_vel, vmin=-10, vmax=10, cmap="RdBu")
    # plt.subplot(1, 6, 5)
    # plt.title("Chip size")
    # plt.imshow(chip_size, vmin=0, vmax=max_chip_size)
    # plt.show()

def preprocess_radar(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image[~np.isfinite(image)] = np.nanmedian(image)
    image = image - image.min()
    return np.clip(image * 3, a_min=1, a_max=255).astype("uint8")

def measure_velocities(
    region: Region, radar_key: str = "ASCENDING_HH", out_res: float = 120.0, min_chip_size: int = 32, max_chip_size: int = 64, debug: bool = False,
) -> Path:
    out_path = Path(f"output/{region.key}/velocity_{radar_key}.zarr")
    if out_path.is_dir():
        return out_path

    data_path = download_region_data(region=region)
        
    with xr.open_zarr(data_path) as all_data:
        if region.key == "bore":
            all_data = all_data.sel(x=slice(473000, 481000), y=slice(8710000, 8701300))
            # all_data["DESCENDING_VV"].isel(time=-1).plot()
            # plt.show()
            # raise NotImplementedError()

        data: xr.DataArray = all_data[radar_key]

        # if data.shape[1] % max_chip_size != 0:
        #     data = data.isel(y=slice(0, data.shape[1] - (data.shape[1] % min_chip_size)))
        # if data.shape[0] % max_chip_size != 0:
        #     data = data.isel(x=slice(0, data.shape[2] - (data.shape[2] % min_chip_size)))

        if data is None:
            raise ValueError("This should be unreachable")

        data = data.sel(time=data.attrs["times"])
        data = data.where(data != 0.)

        data["time_hrs"] = data.time / (1e9 * 3600)
        data["time_hrs"] -= data["time_hrs"].min()

        tree = scipy.spatial.KDTree(data["time_hrs"].values[:, None])
        distances, indices = tree.query(x=data["time_hrs"].values[:, None], k=2, distance_upper_bound=1)
        data["new_time"] = "time", data["time"].values

        times = data["time"].values.copy()
        for i, i2 in enumerate(indices[:, 1]):
            if not np.isfinite(distances[i, 1]):
                continue

            data["new_time"].loc[{"time": times[i]}] = np.mean([times[i], times[i2]])

       
        data = data.groupby("new_time").mean().rename(new_time="time")
        
        #data = data.interpolate_na("x", allow_rechunk=True).interpolate_na("y")
        data = data.ffill("x").ffill("y")
      
        data = data.where((np.abs(data).sum(["x", "y"]) > 100).compute() & (~data.isnull().all("time").compute()), drop=True)

        # data -= data.min("time")
        # data = (data * 3).clip(min=1, max=255).astype("uint8")

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=UserWarning)
        #     data["time"] = data["time"].astype("datetime64[ms]")

        res = data.x.diff("x").isel(x=0).item()

        x_grid = np.arange(0, data.x.shape[0], step=int(out_res / res))
        y_grid = np.arange(0, data.y.shape[0], step=int(out_res / res))

        all_x_coords = data.x.values
        all_y_coords = data.y.values

        y_grid, x_grid = np.meshgrid(y_grid, x_grid)

        xr_coords = []
        velocities: None | xr.Dataset = None
        # The array after is the "arr_before" in the next loop, so keep that instead of reloading it.
        arr_after: None | np.ndarray = None
        for i in tqdm(range(data.time.shape[0] - 1), desc=f"{region.key}: Measuring velocity for {radar_key}"):
            data_before = data.isel(time=i)
            data_after = data.isel(time=i + 1)

            # plt.subplot(2, 1, 1)
            # plt.imshow(data_before)
            # plt.subplot(2, 1, 2)
            # plt.imshow(data_after)
            # plt.show()

            time_before = pd.Timestamp(data_before["time"].item() * 1e6)
            time_after = pd.Timestamp(data_after["time"].item() * 1e6)

            print(f"Measuring {time_before} to {time_after}")

            # Assign the common array from the last iteration, if available
            if arr_after is not None:
                arr_before = arr_after
            else:
                arr_before = preprocess_radar(data_before.values.squeeze())

            arr_after = preprocess_radar(data_after.values.squeeze())

            if debug:
                print(f"{datetime.datetime.now()} Starting autorift")
            rift = autoRIFT()

            rift.I1 = arr_before
            rift.I2 = arr_after

            days_between_acq = (time_after - time_before).total_seconds() / (3600 * 24)
            mid_time = time_before + (time_after - time_before) / 2

            rift.DataType = 0
            rift.xGrid = x_grid
            rift.yGrid = y_grid

            rift.ChipSizeMaxX = max_chip_size
            rift.ChipSizeMinX = min_chip_size
            rift.ChipSize0X = min_chip_size

            # rift.preprocess_filt_hps()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                rift.uniform_data_type()
                rift.runAutorift()

            if debug:
                print(f"{datetime.datetime.now()} Finished autorift")

            if len(xr_coords) == 0:
                xr_coords = [("y", all_y_coords[rift.yGrid[0, :].astype(int)]), ("x", all_x_coords[rift.xGrid[:, 0].astype(int)])]

            if rift.Dx is None or rift.Dy is None or rift.ChipSizeX is None or rift.InterpMask is None:
                raise ValueError("autoRIFT failed (generated no output)")

            x_vel = rift.Dx.T * res / days_between_acq
            y_vel = rift.Dy.T * res / days_between_acq

            out = xr.DataArray(
                x_vel[None, :, :],
                coords=[("time", [mid_time.to_datetime64()])] + xr_coords,
                name="vx",
            ).to_dataset()

            for name, arr in [("vy", y_vel), ("chip_size", rift.ChipSizeX.T), ("interp_mask", rift.InterpMask.T)]:
                out[name] = ("time", "y", "x"), arr[None, :, :]
            out["before_time"] = ("time"), [time_before.to_datetime64()]
            out["after_time"] = ("time"), [time_after.to_datetime64()]

            if debug:
                n_plots = 5

                plt.figure(figsize=(5 * n_plots, 5))
                i = 1
                plt.subplot(1, n_plots, i)
                plt.title(str(time_before))
                plt.imshow(arr_before, vmin=0, vmax=255, cmap="Greys")
                i += 1
                plt.subplot(1, n_plots, i)
                plt.imshow(x_vel, vmin=0, vmax=10, cmap="Reds")
                i += 1
                plt.subplot(1, n_plots, i)
                plt.imshow(y_vel, vmin=0, vmax=10, cmap="Reds")
                i += 1
                plt.subplot(1, n_plots, i)
                plt.title(str(time_after))
                plt.imshow(arr_after, vmin=0, vmax=255, cmap="Greys")

                plt.show()
                

            if velocities is None:
                velocities = out
            else:
                velocities = xr.concat([velocities, out], dim="time")

    if velocities is None:
        raise ValueError("No velocity maps generated. Probably no data given.")

    write_zarr(velocities, out_path)

    return out_path
        


def write_zarr(data: xr.Dataset, out_path: Path):
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "data.zarr"
        data.to_zarr(temp_path, encoding={k: {"compressor": compressor} for k in data.data_vars})

        shutil.move(temp_path, out_path)


def add_geometry(
    data: xr.Dataset, gis_key: str, dem_path: Path = Path("/media/storage/Erik/Data/NPI/DEMs/S0_DTM5/NP_S0_DTM5.tif"), buffer_size_px: float = 3.,
) -> list[str]:
    gdal.UseExceptions()

    exports: list[str] = []

    res = data.x.diff("x").isel(x=0).item()

    bounds = rio.coords.BoundingBox(
        data.x.min().item() - res / 2,
        data.y.min().item() - res / 2,
        data.x.max().item() + res / 2,
        data.y.max().item() + res / 2,
    )

    transform = rio.transform.from_origin(bounds.left, bounds.top, res, res)

    with rio.open(dem_path) as raster:
        window = rio.windows.from_bounds(*bounds, transform=raster.transform)

        dem_trans = rio.windows.transform(window, raster.transform)

        dem = raster.read(1, window=window, masked=True).filled(0.0)

        dataset: gdal.Dataset = gdal_array.OpenArray(dem)
        dataset.SetGeoTransform(dem_trans.to_gdal())

    out_path = "/vsimem/raster.tif"
    gdal.DEMProcessing(out_path, dataset, "aspect")

    with rio.open(out_path) as raster:
        aspect = raster.read(1, masked=True).filled(0)

    gdal.DEMProcessing(out_path, dataset, "slope")

    with rio.open(out_path) as raster:
        slope = raster.read(1, masked=True).filled(0)

    northness = np.cos(np.deg2rad(aspect)) * np.sin(np.deg2rad(slope))
    eastness = np.sin(np.deg2rad(aspect)) * np.sin(np.deg2rad(slope))

    for name, arr in [("northness", northness), ("eastness", eastness), ("dem", dem)]:

        exports.append(name)
            
        out_arr = np.empty((data.y.shape[0], data.x.shape[0]), dtype="float32")

        rasterio.warp.reproject(
            arr,
            destination=out_arr,
            src_transform=dem_trans,
            dst_trasnform=transform,
            src_crs=rio.CRS.from_epsg(32633),
            dst_crs=rio.CRS.from_epsg(32633),
            resampling=rasterio.warp.Resampling.bilinear,
        )

        data[name] = ("y", "x"), out_arr

    domain = [(gpd.read_file(f"GIS/shapes/{gis_key}/domain.geojson").iloc[0].geometry, 1)]
    centerline = gpd.read_file(f"GIS/shapes/{gis_key}/centerline.geojson").iloc[0].geometry

    centerline_points = []
    for dist in np.arange(1, centerline.length + res, res, dtype="uint32"):
        centerline_points.append((centerline.interpolate(dist).buffer(res * buffer_size_px), dist))

    for name, polygons in [("domain", domain), ("centerline", centerline_points)]:
        exports.append(name)
        out_arr = np.zeros((data.y.shape[0], data.x.shape[0]), dtype="uint32")
        rasterio.features.rasterize(
            polygons,
            out=out_arr,
            transform=transform,
            fill=0,
        )

        data[name] = ("y", "x"), out_arr

    data.attrs["centerline"] = centerline.wkt
    data.attrs["domain"] = domain[0][0].wkt
    data.attrs["bounds"] = list(bounds)

    return exports


def get_x_time_labels(xlim: tuple[float | np.datetime64, ...], unit: str = "ns") -> tuple[list[float], list[str]]:

    match unit:
        case "ns":
            multiplier = 1e3
        case "ms":
            multiplier = 1 
        case _:
            raise NotImplementedError(f"No implementation for unit {unit}")

    
    tmin = pd.Timestamp(xlim[0])
    tmax = pd.Timestamp(xlim[1])
    tick_years = np.arange(tmin.year, tmax.year + 1)
    ticks = []
    labels = []
    for year in tick_years:
        for month in range(1, 13):
            tick = pd.Timestamp(year=year, month=month, day=1).to_datetime64().astype(float) * multiplier

            month_str = "JFMAMJJASOND".lower()[month - 1]
            month_str = str(month)

            if month == 1:
                month_str += f"\n{year}"
            ticks.append(tick)
            labels.append(month_str)

    return ticks, labels

def analyze(vel_path: Path):
    key = vel_path.parent.stem
    pol = vel_path.stem.replace("velocity_", "")

    key_to_gis_translation = {
        "vallakra": "vallakrabreen",
    }
    gis_key = key_to_gis_translation.get(key, key)

    with xr.open_dataset(vel_path, engine="zarr") as data, warnings.catch_warnings():
        data = data.sortby("time")

        if "bounds" not in data.attrs:
            add_geometry(data, gis_key=gis_key)

        if "v" not in data.data_vars:
            data["v"] = (data["vx"] ** 2 + data["vy"] ** 2) ** 0.5

        data.attrs["bounds"] = rio.coords.BoundingBox(*data.attrs["bounds"])

        for attr in ["centerline", "domain"]:
            data.attrs[attr] = shapely.from_wkt(data.attrs[attr])
    
        warnings.simplefilter("ignore", category=RuntimeWarning)

        vel_anim_path = Path(f"output/{key}/animations/{key}_velocity.mp4")

        if not vel_anim_path.is_file():
            vel_frame_path = Path(f"output/{key}/velocity_frames")
            vel_frame_path.mkdir(exist_ok=True)
            for time, time_data in tqdm(data.groupby("time"), total=data.time.shape[0], desc="Generating velocity frames"):

                match key:
                    case "scheele":
                        figsize = (6, 8)
                    case _:
                        figsize = (9, 8)
                plt.figure(figsize=figsize)
                time_str = pd.Timestamp(time).strftime("%Y-%m-%d")
                plt.title(time_str)
                plt.imshow(time_data["v"], extent=(data.attrs["bounds"].left, data.attrs["bounds"].right, data.attrs["bounds"].bottom, data.attrs["bounds"].top), vmin=0, vmax=10, cmap="Reds")

                plt.plot(*data.attrs["centerline"].xy, color="black", label="Centerline")
                plt.plot(*data.attrs["domain"].exterior.xy, color="red", alpha=0.3, label="Domain")
                plt.xlabel("Easting (m)")
                plt.ylabel("Northing (m)")
                cbar = plt.colorbar(aspect=20, fraction=0.05)
                cbar.set_label("Velocity (m)")
                plt.legend(loc="upper left")
                plt.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.95)

                plt.savefig(vel_frame_path / f"vel_{time_str}.jpg")
                plt.close()

            animate(vel_frame_path, vel_anim_path)
           
        out_path = Path(f"output/{key}/figures/{key}_velocity.jpg")

        if not out_path.is_file() or True:
            start_times = {
                "vallakra": "2021-01-01",
                "scheele": "2021-01-01",
                "petermann": "2017-06-20",
            }

            if key in start_times:
                start_time = pd.Timestamp(start_times[key], unit="ns")
                data = data.sel(time=slice(start_time, None))

            quant = data["v"].where(data["centerline"] > 0).quantile([0.7, 0.8, 0.9], dim=["x", "y"])

            pd.DataFrame(quant.values.T, columns=quant.coords["quantile"].values, index=quant.time.values).to_csv(out_path.parent.parent / "velocity_quantiles.csv")
            bins = np.linspace(1, data["centerline"].max().item() + 1, num=50)
            bins_to_use = np.r_[[np.nan], bins + np.diff(bins)[0] / 2]
            data["centerline_binned"] = ("y", "x"), bins_to_use[np.digitize(data["centerline"], bins=bins)]

            centerline = data.groupby(data["centerline_binned"]).median()
            centerline["v_std"] = data["v"].groupby(data["centerline_binned"]).std()
            centerline = centerline.rename(centerline_binned="distance")

            plt.figure(figsize=(12, 6))
            plt.suptitle(key)
            plt.subplot(2, 1, 1)
            plt.fill_between(quant.time.astype(float), quant.isel(quantile=0), quant.isel(quantile=2), alpha=0.3, label="70th to 90th percentile")
            plt.plot(quant.time.astype(float), quant.isel(quantile=1), label="80th percentile")
            plt.ylabel("Velocity (m/d)")
            plt.legend(loc="upper left")

            plt.ylim(0, np.nanmax([np.percentile(quant.isel(quantile=2), 99), quant.isel(quantile=1).max().item() * 1.05]))

            xlim = quant.time.astype(float).min().item(),quant.time.astype(float).max().item() 
            # xlim = plt.gca().get_xlim()
            plt.xticks(*get_x_time_labels(xlim), fontsize=9)
            plt.xlim(xlim)
            plt.grid(alpha=0.3)

            axis = plt.subplot(2, 1, 2)

            plt.pcolormesh(
                *np.meshgrid(centerline["time"].values.astype(float), centerline.distance.values),
                centerline["v"].where(centerline["v_std"] < 5),
                vmin=0,
                vmax=10,
                cmap="Reds"
            )
            # plt.imshow(
            #     centerline["v"].where(centerline["v_std"] < 5),
            #     cmap="Reds",
            #     vmin=0,
            #     vmax=10,
            #     aspect="auto",
            #     extent=(
            #         centerline["time"].min().item(),
            #         centerline["time"].max().item(),
            #         centerline["distance"].max().item(),
            #         centerline["distance"].min().item(),
            #     ),
            # )

            # xlim = plt.gca().get_xlim()

            plt.ylim(plt.gca().get_ylim()[::-1])
            plt.xticks(*get_x_time_labels(xlim))
            if key == "vallakra":
                plt.hlines(4000, *xlim, color="gray", linestyles=":")

            inset = axis.inset_axes([0.01, 0.55, 0.02, 0.4])
            cbar = plt.colorbar(fraction=0.03, cax=inset, ax=axis)
            cbar.set_label("Velocity (m/d)")
            plt.xlabel("Time (month + year)")
            plt.ylabel("Distance from top (m)")

            plt.xlim(xlim)

            plt.tight_layout()

            out_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(out_path, dpi=600)
            plt.show()

        return

        for time, time_data in centerline.groupby("time"):
            plt.title(pd.Timestamp(time))
            plt.fill_between(
                time_data["distance"],
                time_data["v"] - time_data["v_std"],
                time_data["v"] + time_data["v_std"],
                alpha=0.3,
            )
            plt.plot(time_data["distance"], time_data["v"])
            plt.ylim(0, 11)
            plt.show()


def merge_and_prep_series(paths: list[Path], min_day_diff: float = 4.):

    pols = "-".join([path.stem.replace("velocity_", "") for path in paths])

    out_path = paths[0].with_stem(f"velocity_{pols}_merged")
    key = out_path.parent.stem
    key_to_gis_translation = {
        "vallakra": "vallakrabreen",
    }
    gis_key = key_to_gis_translation.get(key, key)

    if out_path.is_dir():
        return out_path

    with xr.open_mfdataset(paths, combine="nested", concat_dim="time", engine="zarr") as data:
        data = data.sortby("time")
        for key in ["time", "after_time", "before_time"]:
            data[key] = data[key].astype(float)

        data["time_days"] = data.time / (1e9 * 3600 * 24)
        data["time_days"] -= data["time_days"].min()

        new_cols = add_geometry(data, gis_key=gis_key)
        data["vy"] = data["vy"] - data["vy"].where(data["domain"] != 1).median(["x", "y"])
        data["vx"] = data["vx"] - data["vx"].where(data["domain"] != 1).median(["x", "y"])

        # If the geometry data are not extracted at this point, they will be duplicated along the time dimension
        geometry_data = data[new_cols]
        data = data.drop(new_cols)

        tree = scipy.spatial.KDTree(data["time_days"].values[:, None])
        distances, indices = tree.query(x=data["time_days"].values[:, None], k=2, distance_upper_bound=min_day_diff)
        data["new_time"] = "time", data["time"].values

        times = data["time"].values.copy()
        for i, i2 in enumerate(indices[:, 1]):
            if not np.isfinite(distances[i, 1]):
                continue

            data["new_time"].loc[{"time": times[i]}] = np.mean([times[i], times[i2]])

       
        data = data.groupby("new_time").mean().rename(new_time="time")
        for key in ["time", "after_time", "before_time"]:
            data[key] = data[key].astype("datetime64[ns]")


        data = xr.merge([data, geometry_data])
        data["v"] = (data["vx"] ** 2 + data["vy"] ** 2) ** 0.5

        write_zarr(data, out_path)

    return out_path



def main() -> None:

    regions = load_regions()

    # region = [region for region in regions if region.key == key][0]

    for region in regions:
        if region.key not in ["bore", "scheele", "vallakra", "natascha", "petermann", "johansen", "sonklar", "etonfront"]:
            continue

        if region.key != "etonfront":
            continue

        # if region.key != "scheele":
        #     continue
        # pols = ["DESCENDING_VV", "ASCENDING_HH"]

        # # The ASCENDING_HH is crazy for some reason
        # if region.key in ["sonklar", "scheele"]:
        #     pols = [pols[0]]

        # paths = [measure_velocities(region=region, radar_key=pol, debug=False) for pol in pols]
        # vel_path = merge_and_prep_series(path)
        vel_path = measure_velocities(region=region, radar_key="ASCENDING_HH", debug=False)

        analyze(vel_path)
           
    # return

    # plot_velocities(vel_path)


if __name__ == "__main__":
    main()
