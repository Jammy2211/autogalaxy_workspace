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
from os import path
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
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
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
    path_prefix=path.join("howtogalaxy", "chapter_2"),
    name="tutorial_5_linear_light_profile",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

analysis = ag.AnalysisImaging(dataset=dataset)

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
The intensities of linear light profiles are not a part of the model parameterization. They therefore cannot be
accessed in the resulting galaxies, as seen in previous tutorials, for example:

`plane = result.max_log_likelihood_plane`
`intensity = plane.galaxies[0].bulge.intensity`

The intensities are also only computed once a fit is performed, as they must first be solved for via linear algebra. 
They are therefore accessible via the `Fit` and `Inversion` objects, for example as a dictionary mapping every
linear light profile (defined above) to the intensity values:
"""
fit = result_linear_light_profile.max_log_likelihood_fit

print(fit.linear_light_profile_intensity_dict)

"""
To extract the `intensity` values of a specific component in the model, we use that component as defined in the
`max_log_likelihood_plane`.
"""
plane = fit.plane

bulge = plane.galaxies[0].bulge
disk = plane.galaxies[0].disk

print(
    f"\n Intensity of bulge (lp_linear.Sersic) = {fit.linear_light_profile_intensity_dict[bulge]}"
)
print(
    f"\n Intensity of disk (lp_linear.Exponential) = {fit.linear_light_profile_intensity_dict[disk]}"
)

"""
A `Plane` where all linear light profile objects are replaced with ordinary light profiles using the solved 
for `intensity` values is also accessible.

For example, the linear light profile `Sersic` of the `bulge` component above has a solved for `intensity` of ~0.75. 

The `Plane` created below instead has an ordinary light profile with an `intensity` of ~0.75.
"""
plane = fit.model_obj_linear_light_profiles_to_light_profiles

print(
    f"Intensity via Plane With Ordinary Light Profiles = {plane.galaxies[0].bulge.intensity}"
)

"""
__Visualization__

Linear light profiles and objects containing them (e.g. galaxies, a plane) cannot be plotted because they do not 
have an `intensity` value.

Therefore, the object created above which replaces all linear light profiles with ordinary light profiles must be
used for visualization:
"""
plane = fit.model_obj_linear_light_profiles_to_light_profiles
plane_plotter = aplt.PlanePlotter(plane=plane, grid=dataset.grid)
plane_plotter.figures_2d(image=True)

galaxy_plotter = aplt.GalaxyPlotter(galaxy=plane.galaxies[0], grid=dataset.grid)
galaxy_plotter.figures_2d(image=True)

"""
__Basis__

We can use many linear light profiles to build a `Basis`. 

For example, below, we make a `Basis` out of 10 elliptical Gaussian linear light profiles which: 

 - All share the same centre and elliptical components.
 - Have `sigma` values which increase following the relation `sigma = 0.0001 + (1.0 * log10(1.0 + i)). 
 
Because `log10(1.0) = 0.0` the first Gaussian `sigma` value is therefore 0.0001, whereas because `log10(10) = 1.0`
the size of the final Gaussian is 1.0. The equation below has therefore been chosen to provide intuition on 
the scale of the Gaussians.
"""
gaussian_list = [
    ag.lp_linear.Gaussian(centre=(0.0, 0.0), ell_comps=(0.0, 0.0)) for _ in range(10)
]

for i, gaussian in enumerate(gaussian_list):
    gaussian.sigma = 0.0001 + (1.0 * np.log10(1.0 + i))

basis = ag.lp_basis.Basis(
    light_profile_list=gaussian_list,
)

"""
One we have a `Basis`, we can treat it like any other light profile in order to create a `Galaxy` and `Plane` and 
use it to fit data.
"""
galaxy = ag.Galaxy(redshift=0.5, bulge=basis)

plane = ag.Plane(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, plane=plane)

"""
By plotting the fit, we see that the `Basis` does a reasonable job at capturing the appearance of the galaxy in the 
data

However, there are also distinct residual features:
 
 - The centre of the image has significant chi-squareds, because the `sigma` scale of the Gaussians were not
 small enough to capture the central emission.
 
 - There is a directional patterns in the residuals emanating from the centre of the data, because the Gaussians
 have a single elliptical unit system whereas the data is made of two elliptical structures (a bulge and disk). 

 - There is also a "ringing" pattern, where circular rings can be seen radially outwards from the data. This is
 caused by Gaussians with alternating positive and negative being used to reconstruct the data, and is a result
 of the `Basis` not perfectly representing the underlying galaxy.

We will address are these deficiencies in the model using a ` Basis` that we fit below.
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

 - The bulge is a superposition of 10 parametric linear `Gaussian` profiles [6 parameters]. 
 - The centres and elliptical components of each family of Gaussians are all linked together.
 - The `sigma` size of the Gaussians increases following a relation sigma = a + b * log10(i), where `i` is the 
 Gaussian index (which runs from 0 -> 9) and `a` and `b` are free parameters.

__Relations__

The model below is composed using relations of the form `y = a + (log10(i+1) + b)`, where the values  of `a` 
and `b` are the non-linear free parameters fitted for by `nautilus`. This is the same relation used in the
simple fitting example above.

For example, if `nautilus` samples a model where `a=0.01` and `b=5.0`, it will use a `Basis` containing 10 Gaussians 
whoses `sigma` values are are follows:

 - gaussian[0] -> sigma = 0.01 + (5.0 * log10(0.0 + 1.0)) = 0.01 + 0.0 = 0.01
 - gaussian[1] -> sigma = 0.01 + (5.0 * log10(1.0 + 1.0)) = 0.01 + 0.301 = 0.311
 - gaussian [...] -> continues with increasing sigma.
 - gaussian[9] -> sigma = 0.01 + (5.0 * log10(9.0 + 1.0)) = 0.01 + 1.0 = 1.01

Again, the relations above are chosen to provide intuition on the scale of the Gaussians. 

Because `a` and `b` are free parameters (as opposed to `sigma` which can assume many values), we are able to 
compose and fit `Basis` objects which can capture very complex light distributions with just N = 5-10 non-linear 
parameters!

Owing to our use of relations, the number of free parameters associated with the bulge is N=6. 
"""
bulge_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
bulge_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussians_bulge = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussians_bulge):
    gaussian.centre = gaussians_bulge[0].centre
    gaussian.ell_comps = gaussians_bulge[0].ell_comps
    gaussian.sigma = bulge_a + (bulge_b * np.log10(i + 1))


bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_bulge,
)

"""
The residuals of the fit above showed us that the galaxy in the data is composed of multiple structures (e.g. a bulge
and disk) which have distinct elliptical coordinates.

We therefore compose a second `Basis` of 10 Gaussians to represent the `disk`. This is parameterized the same as
the `bulge` (e.g. all Gaussians share the same `centre` and `ell_comps`) but is treated as a completely
independent set of parameters.
"""
disk_a = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
disk_b = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

gaussians_disk = af.Collection(af.Model(ag.lp_linear.Gaussian) for _ in range(10))

for i, gaussian in enumerate(gaussians_disk):
    gaussian.centre = gaussians_disk[0].centre
    gaussian.ell_comps = gaussians_disk[0].ell_comps
    gaussian.sigma = disk_a + (disk_b * np.log10(i + 1))

disk = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_disk,
)

"""
We now compose the overall model which uses both sets of 10 Gaussians to represent separately the bulge and disk.

The overall dimensionality of non-linear parameter space is just N=12, which is fairly remarkable if you
think about just how complex the structures are that these two `Basis` of Gaussians can capture!
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
The `info` attribute shows the model, which is a lot longer than we have seen previously, given that is 
composed of 20 Gaussians in total!
"""
print(model.info)

"""
We now fit the model.
"""
search = af.Nautilus(
    path_prefix=path.join("howtogalaxy", "chapter_2"),
    name="tutorial_5_basis",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
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

Note that the result objects `max_log_likelihooD_plane` and `max_log_likelihood_fit` automatically convert
all linear light profiles to ordinary light profiles, including every single one of the 20 Gaussians fitted
above. 

This means we can use them directly to perform the visualization below.
"""
print(result_basis.max_log_likelihood_instance)

plane_plotter = aplt.PlanePlotter(
    plane=result_basis.max_log_likelihood_plane, grid=result_basis.grid
)
plane_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result_basis.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Regularization__

There is one downside to `Basis` functions, we may compose a model with too much freedom. The `Basis` (e.g. our 20
Gaussians) may overfit noise in the data -- something we don't want to happen! 
 
To circumvent this issue, we have the option of adding regularization to a `Basis`. Regularization penalizes
solutions which are not smooth -- it is essentially a prior that says we expect the component the `Basis` represents
(e.g. a bulge or disk) to be smooth, in that its light changes smoothly as a function of radius.

Below, we compare two fits, one without regularization and one with regularization which uses a `coefficient=1.0`,
which is a relatively large value that leads to an overly smooth fit.
"""
gaussian_list = [
    ag.lp_linear.Gaussian(centre=(0.0, 0.0), ell_comps=(0.0, 0.0)) for _ in range(10)
]

for i, gaussian in enumerate(gaussian_list):
    gaussian.sigma = 0.0001 + (1.0 * np.log10(1.0 + i))

basis = ag.lp_basis.Basis(light_profile_list=gaussian_list, regularization=None)

galaxy = ag.Galaxy(redshift=0.5, bulge=basis)

plane = ag.Plane(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, plane=plane)

fit_plotter = aplt.FitImagingPlotter(
    fit=fit,
)
fit_plotter.subplot_fit()


basis = ag.lp_basis.Basis(
    light_profile_list=gaussian_list, regularization=ag.reg.Constant(coefficient=1.0)
)

galaxy = ag.Galaxy(redshift=0.5, bulge=basis)

plane = ag.Plane(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, plane=plane)

fit_plotter = aplt.FitImagingPlotter(
    fit=fit,
)
fit_plotter.subplot_fit()

"""
We can easily extend the model-fit performed above to include regularization, where the `coefficient` parameters 
associated with each `Basis` are included in the `nautilus` non-linear parameter space and fitted for.
"""

bulge = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_bulge,
    regularization=af.Model(ag.reg.Constant),
)

disk = af.Model(
    ag.lp_basis.Basis,
    light_profile_list=gaussians_disk,
    regularization=af.Model(ag.reg.Constant),
)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, which has addition priors now associated with regularization.
"""
print(model.info)

"""
We now fit the model.
"""
search = af.Nautilus(
    path_prefix=path.join("howtogalaxy", "chapter_2"),
    name="tutorial_5_basis_regularization",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=1,
)

"""
__When To Regularize?__

The benefits of applying regularization are: 

- It prevents or reduces the over-fitting of noise in the data. 

- It circumvents the "ringing" effect seen above, where alternating negative and positive linear light 
profiles reconstruct the data.

The downside is it adds extra non-linear parameters to the fit, slowing the analysis down.

Whether you need to apply regularization to your science case is a difficult question. We recommend that 
if you are using `Basis` objects, you begin without regularization to see if the `Basis` looks sufficient to 
fit the data accurately. If effects like the positie / negative ringing occur, you may want to then try
fits including regularization. 

Regularization is applied following a statistics framework, which is described in more detail in chapter 4 
of **HowToGalaxy*.

__Other Basis Functions__

In addition to the Gaussians used in this example, there are a number of other linear light profiles 
implemented in **PyAutoGalaxy** which are designed to be used as basis functions:

 - Shapelets: Shapelets are basis functions with analytic properties that are appropriate for capturing the 
   exponential / disk-like features of a galaxy. They do so over a wide range of scales, and can often represent 
   features in these galaxies that a single Sersic function cannot.
 
An example using shapelets is given at `autogalaxy_workspace/scripts/imaging/modeling/features/shapelets.py`.

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
    light_profile_list=gaussian_list, regularization=ag.reg.Constant(coefficient=1.0)
)
