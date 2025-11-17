"""
Tutorial 5: Linear Profiles
===========================

In the previous tutorial we learned how to balance model complexity with our non-linear search in order to infer
accurate model solutions and avoid failure. We saw how in order to fit a model accurately one may have to
parameterize and fit a simpler model with fewer non-linear parameters, at the expense of fitting the data less
accurately.

It would be desirable if we could make our model have more flexibility enabling it to fit more complex galaxy
structures, but in a way that does not increase (or perhaps even decreases) the number of non-linear parameters.
This would keep the `nautilus` model-fit efficient and accurate.

This is possible using linear light profiles, which solve for their `intensity` parameter via efficient linear
algebra, using a process called an inversion. The inversion always computes `intensity` values that give the best
fit to the data (e.g. they minimize the chi-squared and therefore maximize the likelihood).

This tutorial will first fit a model using two linear light profiles. Because their `intensity` values are solved for
implicitly, this means they are not a dimension of the non-linear parameter space fitted by `nautilus`, therefore
reducing the complexity of parameter space and making the fit faster and more accurate.

This tutorial will then show how many linear light profiles can be combined into a `Basis`, which comes from the term
'basis function'. By combining many linear light profiles models can be composed which are able to fit complex galaxy
structures (e.g. asymmetries, twists) with just N=6-8 non-linear parameters.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import autofit as af

"""
__Initial Setup__

we'll use the same galaxy data as the previous tutorial, where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
__Mask__

we'll create and use a smaller 2.5" `Mask2D` again.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
)

dataset = dataset.apply_mask(mask=mask)

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

"""
When plotted, the galaxy's bulge and disk are clearly visible in the centre of the image.
"""
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Linear Light Profiles__

First, we use a variant of a light profile discussed called a "linear light profile", which is accessed via the
command `ag.lp_linear`. 
 
The `intensity` values of linear light profiles are solved for via linear algebra. We use the `Sersic` 
and `Exponential` linear light profiles, which are identical to the ordinary `Sersic` and `Exponential` 
profiles fitted in previous tutorials, except for their `intensity` parameter now being solved for implicitly.

Because the `intensity` parameter of each light profile is not a free parameter in the model-fit, the dimensionality of 
non-linear parameter space is reduced by 1 for each light profile (in this example, 2). This also removes the 
degeneracies between the `intensity` and other light profile parameters (e.g. `effective_radius`, `sersic_index`), 
making the model-fit more robust.

This is a rare example where we are able to reduce the complexity of parameter space without making the model itself 
any simpler. There is really no downside to using linear light profiles, so I would recommend you adopt them as 
standard for your own model-fits from here on!
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, including the linear light profiles.

Note how the `intensity` is no longer listed and does not have a prior associated with it.
"""
print(model.info)

"""
We now create this search and run it.
"""
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_7_linear_light_profile",
    unique_tag=dataset_name,
    n_live=100,
)

analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Run Time__

For standard light profiles, the log likelihood evaluation time is of order ~0.01 seconds for this dataset.

For linear light profiles, the log likelihood evaluation increases to around ~0.05 seconds per likelihood evaluation.
This is still fast, but it does mean that the fit may take around five times longer to run.

However, because two free parameters have been removed from the model (the `intensity` of the lens bulge and 
source bulge), the total number of likelihood evaluations will reduce. Furthermore, the simpler parameter space
likely means that the fit will take less than 10000 per free parameter to converge. This is aided further
by the reduction in `n_live` to 100.

Fits using standard light profiles and linear light profiles therefore take roughly the same time to run. However,
the simpler parameter space of linear light profiles means that the model-fit is more reliable, less susceptible to
converging to an incorrect solution and scales better if even more light profiles are included in the model.

Run the non-linear search.
"""
print(
    "The non-linear search has begun running - checkout the workspace/output/howtogalaxy/chapter_2/tutorial_5_linear_light_profile"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_linear_light_profile = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the resulting model, which does not display the `intensity` values for each light profile.
"""
print(result_linear_light_profile.info)

"""
__Intensities__

The intensities of linear light profiles are not a part of the model parameterization and therefore are not displayed
in the `model.results` file.

To extract the `intensity` values of a specific component in the model, we use the `max_log_likelihood_galaxies`,
which has already performed the inversion and therefore the galaxy light profiles have their solved for
`intensity`'s associated with them.
"""
galaxies = result_linear_light_profile.max_log_likelihood_galaxies

print(galaxies[0].bulge.intensity)

"""
The `Galaxies` contained in the `max_log_likelihood_fit` also has the solved for `intensity` values:
"""
fit = result_linear_light_profile.max_log_likelihood_fit

galaxies = fit.galaxies

print(galaxies[0].bulge.intensity)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the objects created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
galaxies = result_linear_light_profile.max_log_likelihood_galaxies

galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=dataset.grid)
galaxies_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Basis__

We can use many linear light profiles to build a `Basis`. 

For example, below, we make a `Basis` out of 30 elliptical Gaussian linear light profiles which: 

 - All share the same centre and elliptical components.
 - The `sigma` size of the Gaussians increases in log10 increments.
 
Because `log10(1.0) = 0.0` the first Gaussian `sigma` value is therefore 0.0001, whereas because `log10(10) = 1.0`
the size of the final Gaussian is 1.0. 

The equation below has therefore been chosen to provide intuition on the scale of the Gaussians.
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []

# A list of Gaussian model components whose parameters are customized belows.

gaussian_list = af.Collection(
    af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
)

# Iterate over every Gaussian and customize its parameters.

for i, gaussian in enumerate(gaussian_list):
    gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
    gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
    gaussian.ell_comps = gaussian_list[
        0
    ].ell_comps  # All Gaussians have same elliptical components.
    gaussian.sigma = (
        10 ** log10_sigma_list[i]
    )  # All Gaussian sigmas are fixed to values above.

bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

"""
One we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Galaxies` and 
use it to fit data.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
By plotting the fit, we see that the `Basis` does a reasonable job at capturing the appearance of the lens galaxy.

There are imperfections, but this is because we did not fit the model via a non-linear search in order to determine
the optimal values of the Gaussians in the basis. In particular, the Gaussians above were all spherical, when the
lens galaxy is elliptical. 

We rectify this below, where we use a non-linear search to determine the optimal values of the Gaussians!
"""
fit_plotter = aplt.FitImagingPlotter(
    fit=fit,
)
fit_plotter.subplot_fit()

"""
__Model Fit__

To fit a model using `Basis` functions, the API is very similar to that shown throughout this chapter, using both
the `af.Model()` and `af.Collection()` objects.

In this example we fit a `Basis` model for the bulge where:

 - The bulge is a superposition of 30 parametric linear `Gaussian` profiles [4 parameters]. 
 - The centres and elliptical components of each family of Gaussians are all linked together.
 - The `sigma` size of the Gaussians increases in log10 increments.

The number of free parameters and therefore the dimensionality of the MGe is just N=4.
"""
total_gaussians = 30

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []

# A list of Gaussian model components whose parameters are customized belows.

gaussian_list = af.Collection(
    af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
)

# Iterate over every Gaussian and customize its parameters.

for i, gaussian in enumerate(gaussian_list):
    gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
    gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
    gaussian.ell_comps = gaussian_list[
        0
    ].ell_comps  # All Gaussians have same elliptical components.
    gaussian.sigma = (
        10 ** log10_sigma_list[i]
    )  # All Gaussian sigmas are fixed to values above.

bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

bulge = af.Model(
    ag.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

"""
__Disk MGE__

The residuals of the fit above showed us that the galaxy in the data is composed of multiple structures (e.g. a bulge
and disk) which have distinct elliptical coordinates.

We therefore compose a second `Basis` of 10 Gaussians to represent the `disk`. This is parameterized the same as
the `bulge` (e.g. all Gaussians share the same `centre` and `ell_comps`) but is treated as a completely
independent set of parameters.
"""
total_gaussians = 10

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

disk_gaussian_list = []

# A list of Gaussian model components whose parameters are customized belows.

gaussian_list = af.Collection(
    af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
)

# Iterate over every Gaussian and customize its parameters.

for i, gaussian in enumerate(gaussian_list):
    gaussian.centre.centre_0 = bulge_gaussian_list[
        0
    ].centre_0  # All Gaussians have same y centre as bulge.
    gaussian.centre.centre_1 = bulge_gaussian_list[
        0
    ].centre_1  # All Gaussians have same x centre as bulge.
    gaussian.ell_comps = gaussian_list[
        0
    ].ell_comps  # All Gaussians have same elliptical components.
    gaussian.sigma = (
        10 ** log10_sigma_list[i]
    )  # All Gaussian sigmas are fixed to values above.

disk_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

disk = af.Model(
    ag.lp_basis.Basis,
    profile_list=disk_gaussian_list,
)

"""
We now compose the overall model which uses both sets of Gaussians to represent separately the bulge and disk.

The overall dimensionality of non-linear parameter space is just N=6, which is fairly remarkable if you
think about just how complex the structures are that these two `Basis` of Gaussians can capture!
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
The `info` attribute shows the model, which is a lot longer than we have seen previously, given that is 
composed of many Gaussians!
"""
print(model.info)

"""
We now fit the model, with just `n_live=50` given the simiplicity of parameter space.
"""
search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_7_basis",
    unique_tag=dataset_name,
    n_live=50,
)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtogalaxy/chapter_2/tutorial_5_basis"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_basis = search.fit(model=model, analysis=analysis)

"""
__Result__

The result `info` attribute shows the result, which is again longer than usual given the large number of Gaussians
used in the fit.
"""
print(result_basis.info)

"""
Visualizing the fit shows that we successfully fit the data to the noise level.

Note that the result objects `max_log_likelihood_galaxies` and `max_log_likelihood_fit` automatically convert
all linear light profiles to ordinary light profiles, including every single one of the 20 Gaussians fitted
above. 

This means we can use them directly to perform the visualization below.
"""
print(result_basis.max_log_likelihood_instance)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result_basis.max_log_likelihood_galaxies, grid=result_basis.grids.lp
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result_basis.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Multi Gaussian Expansion Benefits__

Symmetric light profiles (e.g. elliptical Sersics) may leave significant residuals, because they fail to capture
irregular and asymmetric morphological of galaxies (e.g. isophotal twists, an ellipticity which varies radially).
An MGE fully captures these features and can therefore much better represent the emission of complex galaxies.

The MGE model can be composed in a way that has fewer non-linear parameters than an elliptical Sersic. In this example,
a groups of Gaussians is used to represent the `bulge` of the galaxy, which in total correspond to just N=4 non-linear 
parameters (a `bulge` and `disk` comprising two linear Sersics has N=10 parameters).

The MGE model parameterization is also composed such that neither the `intensity` parameters or any of the
parameters controlling the size of the Gaussians (their `sigma` values) are non-linear parameters sampled by Nautilus.
This removes the most significant degeneracies in parameter space, making the model much more reliable and efficient
to fit.

Therefore, not only does an MGE fit more complex galaxy morphologies, it does so using fewer non-linear parameters
in a much simpler non-linear parameter space which has far less significant parameter degeneracies!

__Positive Only Solver__

Many codes which use linear algebra typically rely on a linear algabra solver which allows for positive and negative
values of the solution (e.g. `np.linalg.solve`), because they are computationally fast. 

This is problematic, as it means that negative surface brightnesses values can be computed to represent a galaxy's 
light, which is clearly unphysical. For an MGE, this produces a positive-negative "ringing", where the
Gaussians alternate between large positive and negative values. This is clearly undesirable and unphysical.

**PyAutoGalaxy** (and therefore all examples above) uses a positive only linear algebra solver which has been 
extensively optimized to ensure it is as fast as positive-negative solvers. This ensures that all light profile 
intensities are positive and therefore physical. 

__Other Basis Functions__

In addition to the Gaussians used in this example, there is another basis function implemented in PyAutoGalaxy 
that is commonly used to represent the light of galaxies, called a `Shapelet`. 

Shapelets are basis functions with analytic properties that are appropriate for capturing the  exponential / disk-like 
features of a galaxy. They do so over a wide range of scales, and can often represent features in source galaxies 
that a single Sersic function or MGE cannot.

An example using shapelets is given at `autogalaxy_workspace/scripts/modeling/imaging/features/shapelets.py`.
 
Feel free to experiment with using shapelets as the galaxy by yourself. However they incur higher computational 
overheads than the MGE and include a free parameter which governs the size of the basis functions and therefore source,
slowing down convergence of the non-linear search. We have found that MGEs perform better than shapelets in most 
lens modeling problems. 

If you have a desire to fit sources with even more complex morphologies we recommend you look at how to reconstruct 
sources using pixelizations in the `modeling/features` section or chapter 4 of **HowToGalaxy**.

__Wrap Up__

In this tutorial we described how linearizing light profiles allows us to fit more complex light profiles to
galaxies using fewer non-linear parameters, keeping the fit performed by the non-linear search fast, accurate
and robust.

Perhaps the biggest downside to basis functions is that they are only as good as the features they can capture
in the data. For example, a baiss of Gaussians still assumes that they have a well defined centre, but there are
galaxies which may have multiple components with multiple centres (e.g. many star forming knots) which such a 
basis cannot catprue.

In chapter 4 of **HowToGalaxy** we introduce non-parametric pixelizations, which reconstruct the data in way
that does not make assumptions like a centre and can thus reconstruct even more complex, asymmetric and irregular
galaxy morphologies.
"""
basis = ag.lp_basis.Basis(
    profile_list=gaussian_list, regularization=ag.reg.Constant(coefficient=1.0)
)
