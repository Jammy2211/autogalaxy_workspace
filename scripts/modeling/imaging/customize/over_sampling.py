"""
Settings: Over Sampling
=======================

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

This example demonstrates how to change the over sampling used to compute the surface brightness of every image-pixel,
whereby a higher sub-grid resolution better oversamples the image of the light profile so as to provide a more accurate 
model of its image.

**Benefit**: Higher level of over sampling provide a more accurate estimate of the surface brightness in every image-pixel.
**Downside**: Higher levels of over sampling require longer calculations and higher memory usage.

__Prequisites__

You should read up on over-sampling in more detail via  the `autogalaxy_workspace/*/guides/over_sampling.ipynb`
notebook before using this example to customize the over sampling of your model-fits.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset + Masking__ 

For this sub-grid to be used in the model-fit, we must pass the `settings_dataset` to the `Imaging` object,
which will be created using a `Grid2D` with a `sub-size value` of 4 (instead of the default of 2).
"""
dataset_name = "simple__sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the galaxy.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling Lens Galaxy (Uniform)__

The over sampling of the galaxy is controlled using the `OverSamplingUniform` object, where an adaptive
over sampling grid is used to compute the surface brightness of the lens galaxy such that high levels of over sampling
are used in the central regions of the lens galaxy at (0.0", 0.0").
"""
over_sampling = ag.OverSamplingUniform.from_radial_bins(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.1, 0.3],
    centre_list=[(0.0, 0.0)],
)

"""
__Over Sampling__

We now apply the over sampling to the `Imaging` dataset.
"""
dataset = dataset.apply_over_sampling(
    over_sampling=ag.OverSamplingDataset(uniform=over_sampling)
)  # <----- The over sampling above is used here!

"""
__Model + Search + Analysis__ 

The code below performs the normal steps to set up a model-fit. We omit comments of this code as you should be 
familiar with it and it is not specific to this example!
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=path.join("imaging", "settings"),
    name="over_sampling",
    unique_tag=dataset_name,
)

analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Because the `AnalysisImaging` was passed a `Imaging` with a `sub_size=4` it uses a higher level of over sampling
to fit each model `LightProfile` to the data.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

We can confirm that the `Result`'s grid used an over sampling iterate object.
"""
print(result.grids.uniform.over_sampling)

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finish.
"""
