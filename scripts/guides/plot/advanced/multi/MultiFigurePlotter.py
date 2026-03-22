"""
Plots: Multi Figure Plots
=========================

This example illustrates how to produce multiple separate figures for different datasets, using the
function-based plotting API.

The "multi-figure plotter" concept of adding MatPlot objects together no longer exists in the new API. Instead,
each call to a plotting function produces an independent figure. To visualize multiple datasets side-by-side,
simply call the relevant plotting function for each dataset with a distinct `output_filename` so each figure is
saved separately.

An example of when to use this approach is when two different datasets (e.g. at different wavelengths) are loaded
and we want to visualize the image of each dataset as a separate output file. This is the example we
will use in this example script.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load the multi-wavelength `simple` datasets, which we visualize in this example script.
"""
color_list = ["g", "r"]

pixel_scales_list = [0.08, 0.12]

dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"

dataset_path = Path("dataset") / dataset_type / dataset_label / dataset_name

dataset_list = [
    ag.Imaging.from_fits(
        data_path=Path(dataset_path) / f"{color}_data.fits",
        psf_path=Path(dataset_path) / f"{color}_psf.fits",
        noise_map_path=Path(dataset_path) / f"{color}_noise_map.fits",
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

"""
__Individual Subplots__

Plot a full subplot of each dataset individually using `aplt.subplot_imaging_dataset`.
"""
for dataset in dataset_list:
    aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Multiple Output Files__

To produce a separate figure for each dataset (e.g. its data image), call `aplt.plot_array` for each dataset
with a distinct `output_filename`.

This replaces the old `MultiFigurePlotter` approach of adding MatPlot objects together.
"""
for color, dataset in zip(color_list, dataset_list):
    aplt.plot_array(
        array=dataset.data,
        title=f"Data ({color}-band)",
        output_path=dataset_path,
        output_filename=f"data_{color}",
        output_format="png",
    )

"""
We can do the same for the noise map of each dataset.
"""
for color, dataset in zip(color_list, dataset_list):
    aplt.plot_array(
        array=dataset.noise_map,
        title=f"Noise Map ({color}-band)",
        output_path=dataset_path,
        output_filename=f"noise_map_{color}",
        output_format="png",
    )

"""
__Multi Fits__

We can also output a list of figures to a single `.fits` file, where each image goes in each hdu extension as it is
called.

This interface uses a specific method from autoconf called `hdu_list_for_output_from`, which takes a list of
values and a list of extension names, and returns a `HDUList` object that can be written to a `.fits` file.
"""
from autoconf.fitsable import hdu_list_for_output_from

dataset = dataset_list[0]
image_list = [dataset.data, dataset.noise_map]

hdu_list = hdu_list_for_output_from(
    values_list=[image_list[0].mask.astype("float")] + image_list,
    ext_name_list=["mask"] + ["data", "noise_map"],
    header_dict=dataset.mask.header_dict,
)

hdu_list.writeto("dataset.fits", overwrite=True)

"""
__Wrap Up__

In the new API, each call to a plotting function produces an independent figure. To visualize multiple datasets,
simply iterate over them and call the relevant plotting function for each, using `output_filename` to distinguish
the output files.
"""
