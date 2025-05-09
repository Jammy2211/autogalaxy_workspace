"""
Modeling Features: Linear Light Profiles
========================================

A "linear light profile" is a variant of a standard light profile where the `intensity` parameter is solved for
via linear algebra every time the model is fitted to the data. This uses a process called an "inversion" and it
always computes the `intensity` values that give the best fit to the data (e.g. maximize the likelihood)
given the light profile's other parameters.

Based on the advantages below, we recommended you always use linear light profiles to fit models over standard
light profiles!

__Advantages__

Each light profile's `intensity` parameter is therefore not a free parameter in the model-fit, reducing the
dimensionality of non-linear parameter space by the number of light profiles (in this example by 2 dimensions).

This also removes the degeneracies that occur between the `intensity` and other light profile parameters
(e.g. `effective_radius`, `sersic_index`), which are difficult degeneracies for the non-linear search to map out
accurately. This produces more reliable lens model results and the fit converges in fewer iterations, speeding up the
overall analysis.

The inversion has a relatively small computational cost, thus we reduce the model complexity without much slow-down and
can therefore fit models more reliably and faster!

__Disadvantages__

Althought the computation time of the inversion is small, it is not non-negligable. It is approximately 3-4x slower
than using a standard light profile.

The gains in run times due to the simpler non-linear parameter space therefore are somewhat balanced by the slower
likelihood calculation.

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast.

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's
light, which is clearly unphysical.

**PyAutoGalaxy** uses a positive only linear algebra solver which has been extensively optimized to ensure it is as fast
as positive-negative solvers. This ensures that all light profile intensities are positive and therefore physical.

__Model__

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a linear parametric `Sersic` bulge and `Exponential` disk.

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

Load and plot the galaxy dataset `simple` via .fits files, which we will fit with the model.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the galaxy.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Over Sampling__

Apply adaptive over sampling to ensure the calculation is accurate, you can read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.
"""
over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Model__

We compose our model where in this example:

 - The galaxy's bulge is a linear parametric `Sersic` bulge [6 parameters]. 
 - The galaxy's disk is a linear parametric `Exponential` disk, whose centre is aligned with the bulge [3 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.

Note how both light profiles use linear light profiles, meaning that the `intensity` parameter of both is no longer a 
free parameter in the fit.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the light profiles of the lens and source galaxies do not include an `intensity` parameter.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

In the `start_here.py` example 150 live points (`n_live=100`) were used to sample parameter space. For the linear
light profiles this is reduced to 75, as the simpler parameter space means we need fewer live points to map it out
accurately. This will lead to faster run times.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging", "modeling"),
    name="linear_light_profiles",
    unique_tag=dataset_name,
    n_live=300,
    number_of_cores=4,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the model is fitted to the data.
"""
analysis = ag.AnalysisImaging(
    dataset=dataset, settings_inversion=ag.SettingsInversion(use_w_tilde=False)
)

"""
__Run Time__

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

For linear light profiles, the log likelihood evaluation increases to around ~0.05 seconds per likelihood evaluation.
This is still fast, but it does mean that the fit may take around five times longer to run.

However, because two free parameters have been removed from the model (the `intensity` of the bulge and disk), the 
total number of likelihood evaluations will reduce. Furthermore, the simpler parameter space likely means that the 
fit will take less than 10000 per free parameter to converge. This is aided further by the reduction in `n_live` to 75.

Fits using standard light profiles and linear light profiles therefore take roughly the same time to run. However,
the simpler parameter space of linear light profiles means that the model-fit is more reliable, less susceptible to
converging to an incorrect solution and scales better if even more light profiles are included in the model.
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

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that `intensity` parameters are not inferred by the model-fit.
"""
print(result.info)

"""
We plot the maximum likelihood fit, galaxy images and posteriors inferred via Nautilus.

The galaxy bulge and disk appear similar to those in the data, confirming that the `intensity` values inferred by
the inversion process are accurate.
"""
print(result.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=result.grids.lp
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_galaxies`,
which has already performed the inversion and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
galaxies = result.max_log_likelihood_galaxies

print(galaxies[0].bulge.intensity)

"""
The `Galaxies` contained in the `max_log_likelihood_fit` also has the solved for `intensity` values:
"""
fit = result.max_log_likelihood_fit

galaxies = fit.galaxies

print(galaxies[0].bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the objects created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
galaxies = result.max_log_likelihood_galaxies

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=dataset.grid)
galaxies_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Max Likelihood Inversion__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit`, which contains the
`Inversion` object we need.
"""
inversion = result.max_log_likelihood_fit.inversion

"""
This `Inversion` can be used to plot the reconstructed image of specifically all linear light profiles.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
# inversion_plotter.figures_2d(reconstructed_image=True)

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_galaxies`,
which has already performed the inversion and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
galaxies = result.max_log_likelihood_galaxies

print(galaxies[0].bulge.intensity)

"""
The `Galaxies` contained in the `max_log_likelihood_fit` also has the solved for `intensity` values:
"""
fit = result.max_log_likelihood_fit

galaxies = fit.galaxies

print(galaxies[0].bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the objects created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
galaxies = result.max_log_likelihood_galaxies

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=dataset.grid)
galaxies_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Linear Objects (Internal Source Code)__

An `Inversion` contains all of the linear objects used to reconstruct the data in its `linear_obj_list`. 

This list may include the following objects:

 - `LightProfileLinearObjFuncList`: This object contains lists of linear light profiles and the functionality used
 by them to reconstruct data in an inversion. For example it may only contain a list with a single light profile
 (e.g. `lp_linear.Sersic`) or many light profiles combined in a `Basis` (e.g. `lp_basis.Basis`).

- `Mapper`: The linear objected used by a `Pixelization` to reconstruct data via an `Inversion`, where the `Mapper` 
is specific to the `Pixelization`'s `Mesh` (e.g. a `RectnagularMapper` is used for a `Rectangular` mesh).

In this example, two linear objects were used to fit the data:
 
 - An `Sersic` linear light profile.
 ` A `Basis` containing 5 Gaussians. 
"""
print(inversion.linear_obj_list)

"""
To extract results from an inversion many quantities will come in lists or require that we specific the linear object
we with to use. 

Thus, knowing what linear objects are contained in the `linear_obj_list` and what indexes they correspond to
is important.
"""
print(f"LightProfileLinearObjFuncList (Sersic) = {inversion.linear_obj_list[0]}")
print(f"LightProfileLinearObjFuncList (Basis) = {inversion.linear_obj_list[1]}")

"""
Each of these `LightProfileLinearObjFuncList` objects contains its list of light profiles, which for the
`Sersic` is a single entry whereas for the `Basis` is 5 Gaussians.
"""
print(
    f"Linear Light Profile list (Sersic) = {inversion.linear_obj_list[0].light_profile_list}"
)
print(
    f"Linear Light Profile list (Basis -> x5 Gaussians) = {inversion.linear_obj_list[1].light_profile_list}"
)

"""
__Intensities (Internal Source Code)__

The intensities of linear light profiles are not a part of the model parameterization and therefore cannot be
accessed in the resulting galaxies, as seen in previous tutorials, for example:

galaxies = result.max_log_likelihood_galaxies
intensity = galaxies[0].bulge.intensity

The intensities are also only computed once a fit is performed, as they must first be solved for via linear algebra. 
They are therefore accessible via the `Fit` and `Inversion` objects, for example as a dictionary mapping every
linear light profile (defined above) to the intensity values:
"""
fit = result.max_log_likelihood_fit

print(fit.linear_light_profile_intensity_dict)

"""
To extract the `intensity` values of a specific component in the model, we use that component as defined in the
`max_log_likelihood_galaxies`.
"""
galaxies = fit.galaxies

bulge = galaxies[0].bulge
disk = galaxies[0].disk

print(fit.linear_light_profile_intensity_dict)

print(
    f"\n Intensity of bulge (lp_linear.Sersic) = {fit.linear_light_profile_intensity_dict[bulge]}"
)
print(
    f"\n Intensity of first Gaussian in disk = {fit.linear_light_profile_intensity_dict[disk]}"
)

"""
A `Plane` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible.

For example, the linear light profile `Sersic` of the `bulge` component above has a solved for `intensity` of ~0.75. 

The `Plane` created below instead has an ordinary light profile with an `intensity` of ~0.75.
"""
galaxies = fit.model_obj_linear_light_profiles_to_light_profiles

print(
    f"Intensity via Plane With Ordinary Light Profiles = {galaxies[0].bulge.intensity}"
)

"""
Finish.
"""
