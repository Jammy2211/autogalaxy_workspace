"""
Modeling Features: Extra Galaxies
=================================

There may be extra galaxies nearby the main galaxy, whose emission blends with the main galaxy.

If their emission is significant, and close enough to the main galaxy, we may simply mask it from the data
to ensure it does not impact the model-fit. In this script, we first illustrate how to do this, and outline two
different approaches to masking the emission of these extra galaxies which is appropriate for different models.

Next, we consider a different approach which extends the modeling API to include these extra galaxies in the model-fit.
This includes light profiles for every galaxy which fit and subtract their emission. The centres of each galaxy (e.g. 
their brightest pixels in the data)  are used as the centre of the light and mass profiles of these galaxies, in 
order to reduce model complexity.

The second approach is more complex and computationally expensive, but if the emission of the extra galaxies blends 
significantly with the main galaxy emission, it is the best approach to take.

The script concludes with some advanced approaches to modeling extra galaxies, for example where their light is modeled
using a Multi Gaussian Expansion.

__Data Preparation__

To perform modeling which accounts for extra galaxies, a mask of their emission of list of the centre of each extra
galaxy are used to set up the model-fit. For the example dataset used here, these tasks have already been performed and
the metadata (`mask_extra_galaxies.fits` and `extra_galaxies_centres.json` are already included in results folder.

The tutorial `autogalaxy_workspace/*/data_preparation/imaging/optional/extra_galaxies_centres.py`
describes how to create these centres and output them to a `.json` file.

To mask the emission of extra galaxies and omit them from the fit, a `mask_extra_galaxies.fits` file is required.
The `data_preparation` tutorial `autogalaxy_workspace/*/data_preparation/imaging/optional/mask_extra_galaxies.py`
describes how to create this mask.

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
__Dataset__

Load and plot the dataset `extra_galaxies` via .fits files.
"""
dataset_name = "extra_galaxies"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
Visualization of this dataset shows two galaxies either side of the main galaxy.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

We define a bigger circular mask of 6.0" than the 3.0" masks used in other tutorials, to ensure the extra galaxy's 
emission is included.
"""
mask_main = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask_main)

"""
Lets plot the masked imaging to make sure the extra galaxies are included.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Extra Galaxies Mask__

Our first approach to modeling the extra galaxies is to mask their emission in the data and not include them in the
model itself. 

This is the simplest approach, and is the best approach when the extra galaxies are far enough away from the main galaxy
 that their emission does not blend significantly with the its emission (albeit this can be difficult to know for 
 certain).

We load the `mask_extra_galaxies.fits` from the dataset folder, combine it with the 6.0" circular mask and apply it to
the dataset.
"""
mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=dataset.pixel_scales,
)

mask = mask_main + mask_extra_galaxies

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Extra Galaxies Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

For a new user, the details of over-sampling are not important, therefore just be aware that below we make it so that 
all calculations use an adaptive over sampling scheme which ensures high accuracy and precision.

Crucially, this over sampling is applied at the centre of both extra galaxy, ensuring the light of both are over 
sampled correctly.

Once you are more experienced, you should read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), (1.0, 3.5), (-2.0, -3.5)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
We now perform a model-fit using the standard API, where the extra galaxies are not included in the model.

The mask we have applied ensures the extra galaxies do not impact the fit, and the model-fit returns a good fit to the
galaxy.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search = af.Nautilus(
    path_prefix=path.join("imaging", "features"),
    name="extra_galaxies_simple_mask",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=4,
    iterations_per_update=20000,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
The fit is satisfactory, whereby the emission of the extra galaxies is masked and omitted from the model-fit.

This is the simplest approach to modeling extra galaxies, and the best starting point for a new user, especially if
the extra galaxies are far from the main galaxy and no clear blending between their emission is present.

__Extra Galaxies Noise Scaling__

The extra galaxies mask above removed all image pixels which had `True` values. This removed the pixels from
the fit entirely, meaning that their coordinates were not used when performing ray-tracing. This is analogous to
what the circular masks used throughout the examples does. For a light profile fit, the model is not sensitive to the 
exact coordinates of the galaxy light, so this was a good approach.

For more complex models fits, like those using a pixelization, masking regions of the image in a way that removes 
their image pixels entirely from the fit can produce discontinuities in the pixelixation. This can lead to 
unexpected systematics and unsatisfactory results

In this case, applying the mask in a way where the image pixels are not removed from the fit, but their data and 
noise-map values are scaled such that they contribute negligibly to the fit, is a better approach. 

We illustrate the API for doing this below, and show the subplot imaging where the extra galaxies mask has scaled
the data values to zeros, increasing the noise-map values to large values and in turn made the signal to noise
of its pixels effectively zero.
"""
dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask_extra_galaxies = ag.Mask2D.from_fits(
    file_path=path.join(dataset_path, "mask_extra_galaxies.fits"),
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=0.1, centre=(0.0, 0.0), radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
We do not perform a model-fit using this dataset, as using a mask like this requires that we use a pixelization
to fit the galaxy, which you may not be familiar with yet.

In the `features/pixelization.ipynb` example we perform a fit using this noise scaling scheme and a pixelization,
so check this out if you are interested in how to do this.

__Extra Galaxies Dataset__

We are now going to model the dataset with extra galaxies included in the model, where these galaxies include
both the light and mass profiles of the extra galaxies.

We therefore reload the dataset and apply the 6.0" circular mask to it, but do not use the extra galaxies mask
as the emission of the extra galaxies is included in the model.
"""
dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask_main = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask_main)

"""
__Extra Galaxies Centres__

To set up a model including each extra galaxy with light profiles, we input manually the centres of the extra galaxies.

In principle, a model including the extra galaxiess could be composed without these centres. For example, if there were 
two extra galaxies in the data, we could simply add two additional light and mass profiles into the model. 
The modeling API does support this, but we will not use it in this example.

This is because models where the extra galaxies have free centres are often too complex to fit. It is likely the fit 
will infer an inaccurate model and local maxima, because the parameter space is too complex.

For example, a common problem is that one of the extra galaxy light profiles intended to model a nearby galaxy instead 
recenter itself and act as part of the main galaxy's light distribution.

Therefore, when modeling extra galaxies we input the centre of each, in order to fix their light and mass profile 
centres or set up priors centre around these values.

The `data_preparation` tutorial `autogalaxy_workspace/*/data_preparation/imaging/examples/optional/extra_galaxies_centres.py` 
describes how to create these centres. Using this script they have been output to the `.json` file we load below.
"""
extra_galaxies_centres = ag.Grid2DIrregular(
    ag.from_json(file_path=path.join(dataset_path, "extra_galaxies_centres.json"))
)

print(extra_galaxies_centres)

"""
__Model__ 

Perform the normal steps to set up the main model of the galaxy.

A full description of model composition is provided by the model cookbook: 

https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)


"""
__Extra Galaxies Model__ 

We now use the modeling API to create the model for the extra galaxies.

Currently, the extra galaxies API require that the centres of the light and mass profiles are fixed to the input centres
(but the other parameters of the light and mass profiles remain free). 

Therefore, in this example fits a model where:

 - The galaxy's bulge is a linear parametric `Sersic` [6 parameters].
 
 - The galaxy's disk is a linear parametric `Exponential` [5 parameters].

 - Each extra galaxy's light is a linear parametric `SersicSph` profile with fixed centre [2 extra galaxies x 2 parameters = 5 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=16.

"""
# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:
    extra_galaxy = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.SersicSph,
    )

    extra_galaxy.bulge.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Model:

model = af.Collection(
    galaxies=af.Collection(galaxy=galaxy), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model includes extra galaxies that we defined above.
"""
print(model.info)

"""
__Search + Analysis__ 

The code below performs the normal steps to set up a model-fit.

Given the extra model parameters due to the extra gaxies, we increase the number of live points to 200.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "features"),
    name="extra_galaxies_model",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=4,
    iterations_per_update=20000,
)

analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Run Time__

Adding extra galaxies to the model increases the likelihood evaluation times, because their light profiles need 
their images  evaluated and their mass profiles need their deflection angles computed.

However, these calculations are pretty fast for profiles like `SersicSph` and `IsothermalSph`, so only a small
increase in time is expected.

The bigger hit on run time is due to the extra free parameters, which increases the dimensionality of non-linear
parameter space. This means Nautilus takes longer to converge on the highest likelihood regions of parameter space.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

By plotting the maximum log likelihood `FitImaging` object we can confirm the extra galaxies contribute to the fit.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Checkout `autogalaxy_workspace/*/imaging/results` for a full description of analysing results in **PyAutoGalaxy**.

These examples show how the results API can be extended to investigate extra galaxies in the results.

__Approaches to Extra Galaxies__

We illustrated two extremes of how to prevent the emission of extra galaxies impacting the model-fit:

- **Masking**: We masked the emission of the extra galaxies entirely, such that their light did not impact the fit,
  and ignored their mass entirely.

- **Modeling**: We included the extra galaxies in the model, such that their light and mass profiles were fitted.

There are approach that fall between these two, for example the light profiles could be omitted from the model
by applying an extra galaxies mask, but their mass profiles can still be included via the modeling API. You could also
include just the light profiles and not the mass profiles, or visa versa. You could also make the redshifts of the
extra galaxies free parameters in the model, or provide different light and mass profiles for each galaxy.

Extending the modeling API should be straight forward given the above examples, and if anything is unclear then
checkout the model cookbook: 

https://pyautogalaxy.readthedocs.io/en/latest/general/model_cookbook.html

__Multi Gaussian Expansion__

The most powerful way to model the light of extra galaxies is to use a mutli Gaussian expansion (MGE), which is 
documented in the `autogalaxy_workspace/*/imaging/features/multi_gaussian_expansion.py` example script.

The reasons for this will be expanded upon here in the future, but in brief the MGE can capture light profiles
more complex than Sersic profiles using fewer parameters. It can therefore fit many extra galaxies in a model
without increasing the dimensionality of parameter space significantly.

In fact, if a spherical MGE is used to model the light of the extra galaxies each MGE introduced 0 new free parameters
to the model, assuming the centre is fixed to the input centre and the `intensity` values are solved for via the MGE
linear algebra calculation. Complex observations with many extra galaxies therefore become feasible to model.

__Scaling Relations__

The modeling API has full support for composing the extra galaxies such that their light and or mass follow scaling
relations. For example, you could assume that the mass of the extra galaxies is related to their luminosity via a
constant mass-to-light ratio.

This is currently documented in `autogalaxy_workspace/*/guides/advanced/scaling_relation.ipynb`, but will be
moved here in the near future.

__Wrap Up__

The extra galaxies API makes it straight forward for us to model galaxies with extra galaxy components for
the light and mass of nearby objects.
"""