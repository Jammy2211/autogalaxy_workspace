"""
Database Optional: Manual
=========================

The main tutorials use the built-in PyAutoGalaxy aggregator objects (e.g. `GalaxiesAgg`) to navigate the database. For the
majority of use-cases this should be sufficient, however a user may have a use case where a more customized
generation of a `Plane` or `FitImaging` object is desired.

This optional tutorials shows how one can achieve this, by creating lists and writing your own generator funtions
to make these objects.
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
__Aggregator__

First, set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "results_folder"),
)

"""
__Manual Planes via Lists (Optional)__

I now illustrate how one can create galaxies via lists. This does not offer any new functionality that the `GalaxiesAgg`
object above does not provide, and is here for illustrative purposes. It is therefore optionag.

Lets create a list of instances of the maximum log likelihood models of each fit.
"""
ml_instances = [samps.max_log_likelihood() for samps in agg.values("samples")]

"""
A model instance contains a list of `Galaxy` instances, which is what we are using to passing to functions in 
PyAutoGalaxy. 

Lets create the maximum log likelihood galaxies of every fit.
"""
ml_galaxies = [ag.Galaxies(galaxies=instance.galaxies) for instance in ml_instances]

print("Maximum Log Likelihood Galaxies: \n")
print(ml_galaxies, "\n")
print("Total Planes = ", len(ml_galaxies))

"""
Now lets plot their convergences, using a grid of 100 x 100 pixels (noting that this isn't` necessarily the grid used
to fit the data in the search itself).
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for galaxies in ml_galaxies:
    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(convergence=True)


"""
__Manual Plane via Generators (Optional / Advanced)__

I now illustrate how one can create galaxies via generators. There may be occasions where the functionality of 
the `GalaxiesAgg` object is insufficient to perform the calculation you require. You can therefore write your own 
generator to do this.

This section is optional, and I advise you only follow it if the `GalaxiesAgg` object is sufficient for your use-case.
"""


def make_galaxies_generator(fit):
    samples = fit.value(name="samples")

    return ag.Galaxies(galaxies=samples.max_log_likelihood().galaxies)


"""
We `map` the function above using our aggregator to create a galaxies generator.
"""
galaxies_gen = agg.map(func=make_galaxies_generator)

"""
We can now iterate over our galaxies generator to make the plots we desire.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for galaxies in galaxies_gen:
    galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
    galaxies_plotter.figures_2d(convergence=True, potential=True)


"""
Now lets use a generator to print the Einstein Mass of the galaxies
"""


def print_max_log_likelihood_mass(fit):
    samples = fit.value(name="samples")

    instance = samples.max_log_likelihood()

    galaxies = ag.Galaxies(galaxies=instance.galaxies)

    einstein_mass = galaxies[0].einstein_mass_angular_from(grid=grid)

    print("Einstein Mass (angular units) = ", einstein_mass)

    cosmology = ag.cosmo.Planck15()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=fit.instance.galaxies.galaxy.redshift,
            redshift_1=fit.instance.galaxies.source.redshift,
        )
    )

    einstein_mass_kpc = einstein_mass * critical_surface_density

    print("Einstein Mass (kpc) = ", einstein_mass_kpc)
    print("Einstein Mass (kpc) = ", "{:.4e}".format(einstein_mass_kpc))


print()
print("Maximum Log Likelihood Lens Einstein Masses:")
agg.map(func=print_max_log_likelihood_mass)


"""
__Manual Dataset via List (Optional)__

I now illustrate how one can create fits via lists. This does not offer any new functionality that the `FitImagingAgg`
object above does not provide, and is here for illustrative purposes. It is therefore optionag.

Lets create a list of the imaging dataset of every lens our search fitted. 

The individual masked `data`, `noise_map` and `psf` are stored in the database, as opposed to the `Imaging` object, 
which saves of hard-disk space used. Thus, we need to create the `Imaging` object ourselves to inspect it. 

They are stored as .fits HDU objects, which can be converted to `Array2D` and `Kernel2D` objects via the
`from_primary_hdu` method.
"""
data_gen = agg.values(name="dataset.data")
noise_map_gen = agg.values(name="dataset.noise_map")
psf_gen = agg.values(name="dataset.psf")
settings_dataset_gen = agg.values(name="dataset.settings")

for data, noise_map, psf, settings_dataset in zip(
    data_gen, noise_map_gen, psf_gen, settings_dataset_gen
):
    data = ag.Array2D.from_primary_hdu(primary_hdu=data)
    noise_map = ag.Array2D.from_primary_hdu(primary_hdu=noise_map)
    psf = ag.Kernel2D.from_primary_hdu(primary_hdu=psf)

    dataset = ag.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_dataset,
        pad_for_convolver=True,
        check_noise_map=False,
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Manual Fit via Generators (Optional / Advanced)__

I now illustrate how one can create fits via generators. There may be occasions where the functionality of 
the `FitImagingAgg` object is insufficient to perform the calculation you require. You can therefore write your own 
generator to do this.

This section is optional, and I advise you only follow it if the `FitImagingAgg` object is sufficient for your use-case.
"""


def make_imaging_gen(fit):
    data = ag.Array2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.data"))
    noise_map = ag.Array2D.from_primary_hdu(
        primary_hdu=fit.value(name="dataset.noise_map")
    )
    psf = ag.Kernel2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.psf"))
    settings_dataset = fit.value(name="dataset.settings")

    dataset = ag.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_dataset,
        pad_for_convolver=True,
        check_noise_map=False,
    )

    return dataset


imaging_gen = agg.map(func=make_imaging_gen)

for dataset in imaging_gen:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()


"""
We now have access to the `Imaging` data we used to perform a model-fit, and the results of that model-fit in the form
of a `Samples` object. 

We can therefore use the database to create a `FitImaging` of the maximum log-likelihood model of every model to its
corresponding dataset, via the following generator:
"""


def make_fit_imaging_generator(fit):
    dataset = make_imaging_gen(fit=fit)

    galaxies = ag.Galaxies(galaxies=fit.instance.galaxies)

    return ag.FitImaging(dataset=dataset, galaxies=galaxies)


fit_imaging_gen = agg.map(func=make_fit_imaging_generator)

for fit in fit_imaging_gen:
    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()

"""
The `AnalysisImaging` object has a `settings_inversion` attributes, which customizes how the inversion fits the 
data. The generator above uses the `settings` of the object that were used by the model-fit. 

These settings objected are contained in the database and can therefore also be passed to the `FitImaging`.
"""


def make_fit_imaging_generator(fit):
    dataset = make_imaging_gen(fit=fit)

    settings_inversion = fit.value(name="settings_inversion")

    galaxies = ag.Galaxies(galaxies=fit.instance.galaxies)

    return ag.FitImaging(
        dataset=dataset,
        galaxies=galaxies,
        settings_inversion=settings_inversion,
    )


fit_imaging_gen = agg.map(func=make_fit_imaging_generator)

for fit in fit_imaging_gen:
    fit_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_plotter.subplot_fit()


"""
__Errors: Axis Ratio__

To begin, lets compute the axis ratio of a model, including the errors on the axis ratio. In the previous tutorials, 
we saw that the errors on a quantity like the ell_comps is simple, because it was sampled by the non-linear 
search. Thus, to get their we can uses the Samples object to simply marginalize over all over parameters via the 1D 
Probability Density Function (PDF).

But what if we want the errors on the axis-ratio? This wasn`t a free parameter in our model so we can`t just 
marginalize over all other parameters.

Instead, we need to compute the axis-ratio of every model sampled by the non-linear search and from this determine 
the PDF of the axis-ratio. When combining the different axis-ratios we weight each value by its `weight`. For Nautilus,
the nested sampler we fitted our aggregator sample with, this down weight_list the model which gave lower likelihood 
fits. For other non-linear search methods (e.g. MCMC) the weight_list can take on a different meaning but can still be 
used for combining different model results.

Below, we get an instance of every Nautilus sample using the `Samples`, compute that models axis-ratio, store them in a 
list and find the value via the PDF and quantile method.

Now, we iterate over each Samples object, using every model instance to compute its axis-ratio. We combine these 
axis-ratios with the samples weight_list to give us the weighted mean axis-ratio and error.

To do this, we again use a generator. Whislt the axis-ratio is a fairly light-weight value, and this could be
performed using a list without crippling your comptuer`s memory, for other quantities this is not the case. Thus, for
computing derived quantities it is good practise to always use a generator.

[Commented out but should work fine if you uncomment it]
"""

#
# def axis_ratio_error_from_agg_obj(fit):
#     samples = fit.value(name="samples")
#
#     axis_ratio_list = []
#     weight_list = []
#
#     for sample_index in range(samples.total_samples):
#         weight = samples.sample_list[sample_index].weight
#
#         if weight > 1e-4:
#             instance = samples.from_sample_index(sample_index=sample_index)
#
#             axis_ratio = ag.convert.axis_ratio_from(
#                 ell_comps=instance.galaxies.galaxy.bulge.ell_comps
#             )
#
#             axis_ratio_list.append(axis_ratio)
#             weight_list.append(weight)
#
#     median_axis_ratio, lower_axis_ratio, upper_axis_ratio = af.marginalize(
#         parameter_list=axis_ratio_list, sigma=3.0, weight_list=weight_list
#     )
#
#     return median_axis_ratio, lower_axis_ratio, upper_axis_ratio
#
#
# axis_ratio_values = list(agg.map(func=axis_ratio_error_from_agg_obj))
# median_axis_ratio_list = [value[0] for value in axis_ratio_values]
# lower_axis_ratio_list = [value[1] for value in axis_ratio_values]
# upper_axis_ratio_list = [value[2] for value in axis_ratio_values]
#
# print("Axis Ratios:")
# print(median_axis_ratio_list)
#
# print("Axis Ratio Errors:")
# print(lower_axis_ratio_list)
# print(upper_axis_ratio_list)

"""
Fin.
"""
