"""
Interferometer: Data Preparation
================================

When an interferometer dataset is analysed, it must conform to certain standards in order for
the analysis to be performed correctly. This tutorial describes these standards and links to more detailed scripts
which will help you prepare your dataset to adhere to them if it does not already.

__SLACK__

The interferometer data preparation scripts are currently being developed and are not yet complete. If you are 
unsure of how to prepare your dataset, please message us on Slack and we will help you directly!

__Pixel Scale__

When fitting an interferometer dataset, the images of the lens  galaxies are first evaluated in real-space
using a grid of pixels, which is then Fourier transformed to the uv-plane.

The "pixel_scale" of an interferometer dataset is this pixel-units to arcsecond-units conversion factor. The value
depends on the instrument used to observe the lens, the wavelength of the light used to observe it and size of
the baselines used (e.g. longer baselines means higher resolution and therefore a smaller pixel scale).

The pixel scale of some common interferometers is as follows:

 - ALMA: 0.02" - 0.1" / pixel
 - SMA: 0.05" - 0.1" / pixel
 - JVLA: 0.005" - 0.01" / pixel

It is absolutely vital you use a sufficently small pixel scale that all structure in the data is resolved after the
Fourier transform. If the pixel scale is too large, the Fourier transform will smear out the data and the model.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

# %matplotlib inline
from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

dataset_path = path.join("dataset", "interferometer", "simple")

"""
__Visibilities__

The image is the image of your galaxy, which comes from a telescope like the Hubble Space telescope (HST).

Lets inspect an image which conforms to **PyAutoGalaxy** standards:
"""
visibilities = ag.Visibilities.from_fits(
    file_path=path.join(dataset_path, "data.fits"), hdu=0
)

array_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
array_plotter.figure_2d()

"""
These visibilities conforms to **PyAutoGalaxy** standards, because they come from a standard CASA data reduction
procedure. 

More details of this procedure are given in the `examples/case_to_autogalaxy.ipynb` notebook.

__Noise-Map__

The noise-map is the real and complex noise in each visiblity of the interferometer dataset. It is used to weight
the visibilities when a model is fitted to the data via a chi-squared statistic.

It is common for all visibilities to have the same noise value, depending on the instrument used to observe the
the data.
"""
visibilities = ag.VisibilitiesNoiseMap.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), hdu=0
)

array_plotter = aplt.Grid2DPlotter(grid=visibilities.in_grid)
array_plotter.figure_2d()

"""
__UV Wavelengths__

The uv-wavelengths define the baselines of the interferometer. They are used to Fourier transform the image to the
uv-plane, which is where the model is evaluated.
"""
uv_wavelengths = ag.util.array_2d.numpy_array_2d_via_fits_from(
    file_path=path.join(dataset_path, "uv_wavelengths.fits"), hdu=0
)

uv_wavelengths = ag.Grid2DIrregular.from_yx_1d(
    y=uv_wavelengths[:, 1] / 10**3.0,
    x=uv_wavelengths[:, 0] / 10**3.0,
)

grid_plotter = aplt.Grid2DPlotter(grid=uv_wavelengths)
grid_plotter.figure_2d()

"""
These uv wavelengths conform to **PyAutoGalaxy** standards, because they come from a standard CASA data reduction
procedure. 

More details of this procedure are given in the `examples/case_to_autogalaxy.ipynb` notebook.

__Real Space Mask__

The `modeling` scripts also define a real-space mask, which defines where the image is evalated in real space 
before it is Fourier transformed.

You must double check that the real-space mask you use:
 
 - Spatially covers the lensed source galaxy, such that the source is not truncated by the mask.
 - Is high enough resolution that the lensed source galaxy is not smeared via the Fourier transform.
 
__Profiling__

If you are analysing an interfeometer dataset with many visibilities (e.g. 1 million and above) and a high 
resolution real-space mask (e.g. 0.01" / pixel), the analysis can take a long time to run. 

The `examples/profiling.ipynb` script shows how to profile and setup your analysis to ensure it have fast enough
run times.

__Data Processing Complete__

If your visibilities, noise-map, uv_wavelengths and real space mask conform the standards above, you are ready to analyse 
your dataset!

Below, we provide an overview of optional data preparation steps which prepare other aspects of the analysis. 

New users are recommended to skim-read the optional steps below so they are aware of them, but to not perform them 
and instead analyse their dataset now. You can come back to the data preparation scripts below if it becomes necessary.

The following scripts are used to prepare components of an interferometer dataset, however they are used in an
identical fashion for dataset datasets.

Therefore, they are not located in the `interferometer/data_preparation` package, but instead in the
`imaging/data_preparation` package, so refer there for a description of their usage.

Note that in order to perform some tasks (e.g. mark on the image where the source is), you will need to use an image
of the interferometer data even though visibilities are used for the analysis.

__Positions (Optional)__

The script allows you to mark the (y,x) arc-second positions of the multiply imaged lensed source galaxy in 
the image-plane, under the assumption that they originate from the same location in the source-plane.

A non-linear search (e.g. nautilus) can then use these positions to preferentially choose mass models where these 
positions trace close to one another in the source-plane. This speeding up the initial fitting of models and 
removes unwanted solutions from parameter space which have too much or too little mass in the galaxy.

If you create positions for your dataset, you must also update your modeling script to use them by loading them 
and passing them to the `Analysis` object via a `PositionsLH` object. 

If your **PyAutoGalaxy** analysis is struggling to converge to a good model, you should consider using positions
to help the non-linear search find a good model.

Links / Resources:

Position-based model resampling is particularly important for fitting pixelized source models, for the
reasons disucssed in the following readthedocs 
webapge  https://pyautogalaxy.readthedocs.io/en/latest/general/demagnified_solutions.html

The script `data_prepration/gui/positions.ipynb` shows how to use a Graphical User Interface (GUI) to mask the 
positions on the lensed source.

See `autogalaxy_workspace/*/interferometer/modeling/customize/positions.py` for an example.of how to use positions in a 
`modeling` script.

__Lens Light Centre (Optional)__

This script allows you to mark the (y,x) arcsecond locations of the galaxy light centre(s) of the galaxy
you are analysing. These can be used as fixed values for the lens light and mass models in a model-fit.

This  reduces the number of free parameters fitted for in a model and removes inaccurate solutions where
the lens mass model centre is unrealistically far from its true centre.

Advanced `chaining` scripts often use these input centres in the early fits to infer an accurate initial model,
amd then make the centres free parameters in later searches to ensure a general and accurate model is inferred.

If you create a `light_centre` for your dataset, you must also update your modeling script to use them.

If your **PyAutoGalaxy** analysis is struggling to converge to a good model, you should consider using a fixed
lens light and / or mass centre to help the non-linear search find a good model.

Links / Resources:

The script `data_prepration/gui/light_centre.ipynb` shows how to use a Graphical User Interface (GUI) to mask the
galaxy light centres.

__Clumps (Optional)__

There may be galaxies nearby the lens  galaxies, whose emission blends with that of the lens 
and whose mass may contribute to the ray-tracing and model.

We can include these galaxies in the model, either as light profiles, mass profiles, or both, using the
**PyAutoGalaxy** clump API, where these nearby objects are given the term `clumps`.

This script marks the (y,x) arcsecond locations of these clumps, so that when they are included in the model the
centre of these clumps light and / or mass profiles are fixed to these values (or their priors are initialized
surrounding these centres).

The example `scaled_dataset.py` (see below) marks the regions of an image where clumps are present, but  but instead 
remove their signal and increase their noise to make them not impact the fit. Which approach you use to account for 
clumps depends on how significant the blending of their emission is and whether they are expected to impact the 
ray-tracing.

This tutorial closely mirrors tutorial 7, `light_centre`, where the main purpose of this script is to mark the
centres of every object we'll model as a clump. A GUI is also available to do this.

Links / Resources:

The script `data_prepration/gui/clump_centres.ipynb` shows how to use a Graphical User Interface (GUI) to mark the
clump centres in this way.

The script `modeling/features/clumps.py` shows how to use clumps in a model-fit, including loading the clump centres
created by this script.

__Info (Optional)__

Auxiliary information about a galaxy dataset may used during an analysis or afterwards when interpreting the 
modeling results. For example, the redshifts of the source and galaxy. 

By storing these as an `info.json` file in the lens's dataset folder, it is straight forward to load the redshifts 
in a modeling script and pass them to a fit, such that **PyAutoGalaxy** can then output results in physical 
units (e.g. kpc instead of arc-seconds).

For analysing large quantities of modeling results, **PyAutoGalaxy** has an sqlite database feature. The info file 
may can also be loaded by the database after a model-fit has completed, such that when one is interpreting
the results of a model fit additional data on a lens can be used to. 

For example, to plot the model-results against other measurements of a lens not made by PyAutoGalaxy. Examples of such 
data might be:

- The velocity dispersion of the galaxy.
- The stellar mass of the galaxy.
- The results of previous galaxy models to the lens performed in previous papers.
"""
