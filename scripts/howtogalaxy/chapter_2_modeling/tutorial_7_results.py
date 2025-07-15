"""
Tutorial 5: Results
===================

In the previous tutorials, each search returned a `Result` object, which we used to plot the maximum log likelihood
fit each model-fit. In this tutorial, we'll take a look at the result object in a little more detail.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt
import autofit as af

"""
__Initial Setup__

Lets use the model-fit performed in tutorial 1 to get a `Result` object.
"""
dataset_name = "simple__sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

model = af.Collection(
    galaxies=af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5, mass=ag.lp.Sersic))
)

search = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_2"),
    name="tutorial_1_non_linear_search",
    unique_tag=dataset_name,
    n_live=80,
)

analysis = ag.AnalysisImaging(dataset=dataset)

result = search.fit(model=model, analysis=analysis)

"""
__Galaxies & Fit__

In the previous tutorials, we saw that this result contains the maximum log likelihood fit, which provide
a fast way to visualize the result.

It also contains the maximum log likelihood galaxies.
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=mask.derive_grid.all_false
)
galaxies_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Samples__

The result contains a lot more information about the model-fit. 

For example, the `Samples` object contains all of the non-linear search samples, including the parameters 
of every successful model evaluation and their log likelihood values. These are used for computing information 
about the model-fit, such as the most probably parameter estimates and the error inferred for every parameter.
"""
print(result.samples)
print("Parameters of 1st Sample:")
print(result.samples.parameter_lists[0][:])
print("Log Likelihood of 1st Sample:")
print(result.samples.log_likelihood_list[0])

"""
__Workspace__

We are not going into any more detail on the result variable in this tutorial, or in the **HowToGalaxy**  lectures.

A comprehensive description of the results can be found at the following packages:

 `autogalaxy_workspace/*/results`
 
The results API for CCD imaging data is the same as for other data types (e.g. interferometer, point-soures). This
package can therefore be used to learn the API and then translate to other data types.

__Database__

Once a search has completed running, we have a set of results on our hard disk which we can manually inspect and 
analyse. Alternatively, we can return the results from the search.fit() method and manipulate them in a Python script
or notebook.  

However, imagine you have a large dataset consisting of many images of galaxies. You analyse each image 
individually using a search, producing a large library of results on your hard disk. There will be lots of paths and 
directories to navigate, and at some point there will simply be too many results for it to be an efficient use of your 
time to analyse the results by sifting through the outputs on your hard disk one-by-one.

**PyAutoGalaxy**'s database tools solve this problem, by making it possible for us to write the results to a .sqlite 
database file and load the results from hard-disk to a Python script or Jupyter notebook. This database supports
advanced queries, so specific results can be loaded and inspected.

We won't go into any more detail on the database in this tutorial. If you think the database will be useful, checkout 
the full set of database tutorials which can be found in the folder `autogalaxy_workspace/*/imaging/advanced/database`. 

Here, you'll learn how to:

 - Use the database to query for results which fit a certain model or give a certain result. 
 
 - Use the `Samples` to produce many different results from the fit, including error estimates on parameters and 
 plots of the probability density function of parameters in 1D and 2D.
 
 - Visualize results, for example the fit to a lens dataset.


__Wrap Up__

Even if you are only modeling a small sample of galaxies, if you anticipate using **PyAutoGalaxy** for the long-term I 
strongly recommend you begin using the database to inspect and analyse your result. 

This is because it makes it simple to perform all analyse in a Jupyter notebook, which is the most flexible and 
versatile way to check results and make figures.
"""
