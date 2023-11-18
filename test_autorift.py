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


def plot_velocities(velocity_data_path: Path):

    
    with xr.open_dataset(velocity_data_path, engine="zarr") as data:

        data["vy"] = data["vy"] - data["vy"].median(["x", "y"])
        data["vx"] = data["vx"] - data["vx"].median(["x", "y"])
        data["v"] = (data["vx"] ** 2 + data["vy"] ** 2) ** 0.5


        # Scheelebreen
        #point = data.sel(x=547940, y=8631397, method="nearest")

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

def measure_velocities(key: str = "scheele", days_between_acq = 12, out_res: float = 120., min_chip_size: int = 32, max_chip_size: int = 64):

    out_path = Path(f"output/{key}/velocity_ASC_HH.zarr")

    if out_path.is_dir():
        return out_path

    all_image_paths = list(Path(f"output/{key}/vrts/ASCENDING_HH").glob("*.vrt"))
    all_image_paths.sort(reverse=True)

    image_paths = {"I2": all_image_paths[0]}

    with rio.open(image_paths["I2"]) as raster:

        shape = (raster.height, raster.width)
        resolution = raster.res[0]
        bounds = raster.bounds

    x_grid = np.arange(0, shape[1], step=int(out_res / resolution))
    y_grid = np.arange(0, shape[0], step=int(out_res / resolution))

    x_coords = np.linspace(bounds.left + resolution / 2, bounds.right - resolution / 2, shape[1])[x_grid]
    y_coords = np.linspace(bounds.bottom + resolution / 2, bounds.top - resolution / 2, shape[0])[::-1][y_grid]

    xr_coords = [("y", y_coords), ("x", x_coords)]

    y_grid, x_grid = np.meshgrid(y_grid, x_grid)

    data: None | xr.Dataset = None

    for i in tqdm(range(len(all_image_paths) - 1)):
        rift = autoRIFT()
        image_paths = {
            "I1": all_image_paths[i + 1],
            "I2": all_image_paths[i]
        }

        time_before = pd.to_datetime(image_paths["I1"].stem.split("_")[-1], format="%Y-%m-%dT%H-%M-%S", utc=True)
        time_after = pd.to_datetime(image_paths["I2"].stem.split("_")[-1], format="%Y-%m-%dT%H-%M-%S", utc=True)

        mid_time = time_before + (time_after - time_before) / 2

        images = {}
        for key in image_paths:
            with rio.open(image_paths[key]) as raster:
                image = raster.read(1, masked=True).filled(0)
                image -= image.min()
                #image = image[500:1500, 350:750]

                images[key] = image
                images[key] = np.clip(image * 3, a_min=1, a_max=255).astype("uint8")

                setattr(rift, key, images[key])

        rift.DataType = 0

        rift.xGrid = x_grid
        rift.yGrid = y_grid

        rift.ChipSizeMaxX = max_chip_size
        rift.ChipSizeMinX = min_chip_size
        rift.ChipSize0X = min_chip_size

        #rift.preprocess_filt_hps()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rift.uniform_data_type()
            rift.runAutorift()

        x_vel = rift.Dx.T * resolution / days_between_acq
        y_vel = rift.Dy.T * resolution / days_between_acq
        chip_size = rift.ChipSizeX.T


        out = xr.DataArray(
            x_vel[None, :, :],
            coords=[("time", [mid_time.to_datetime64()])] + xr_coords,
            name="vx",
        ).to_dataset()
        out["vy"] = ("time", "y", "x"), y_vel[None, :, :]
        out["chip_size"] = ("time", "y", "x"), chip_size[None, :, :].astype("uint8")
        out["interp_mask"] = ("time", "y", "x"), rift.InterpMask.T[None, :, :].astype("uint8")
        out["before_time"] = ("time"), [time_before.to_datetime64()]
        out["after_time"] = ("time"), [time_after.to_datetime64()]

        if data is None:
            data = out
        else:
            data = xr.concat([data, out], dim="time")

    if data is None:
        return

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "vel.zarr"
        #data.to_netcdf(temp_path, encoding={k: {"zlib": True, "complevel": 5} for k in data.data_vars})
        data.to_zarr(temp_path, encoding={k: {"compressor": compressor} for k in data.data_vars}) 

        shutil.move(temp_path, out_path)

    return out_path

        
        

def main():
    vel_path = measure_velocities("vallakra")

    if vel_path is None:
        return
    
    plot_velocities(vel_path)



if __name__ == "__main__":
    main()
