"""
Tutorial 1: Search Chaining
===========================

In chapter 2, we learnt how to perform modeling using a non-linear search. In all of the tutorials, we fitted the
data using just one non-linear search. In this chapter, we introduce a technique called 'non-linear search chaining',
which fits a model using a sequence of non-linear searches. The initial searches fit simpler models whose parameter
spaces can be more accurately and efficiently sampled. The results of this search are then passed to later searches
which fit models of gradually increasing complexity.

Lets think back to tutorial 4 of chapter 2. We learnt there were three approaches one could take fitting a model
accurately if we found that a model fit failed. These were:

 1) Tuning our priors to the galaxy we're fitting.
 2) Making our model less complex.
 3) Searching non-linear parameter space for longer.

However, each of the above approaches has disadvantages. The more we tune our priors, the less we can generalize our
analysis to a different galaxy. The less complex we make our model, the less realistic it is. And if we rely too
much on searching parameter space for longer, we could end up with search`s that take days, weeks or months to run.

In this tutorial, we are going to show how search chaining combines these 3 approaches such that we can fit
complex and realistic models in a way that that can be generalized to many different galaxies. To do this,
we'll run 2 searches, and chain the model inferred in the first search to the priors of the second search`s lens
model.

Our first search will make the same bulge-disk alignment assumption we made in the previous tutorial. We saw that this
gives a reasonable model. However, we'll make a couple of extra simplifying assumptions, to really try and bring
our model complexity down and get the non-linear search running fast.

The model we infer above will therefore be a lot less realistic. But it does not matter, because in the second search
we are going to relax these assumptions and fit the more realistic model. The beauty is that, by running the first
search, we can use its results to tune the priors of our second search. For example:

 1) The first search should give us a pretty good idea of the galaxy's bulge and disk profiles, for example its
 centre, intensity, effective radius.
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

we'll use the same galaxy data as tutorial 4 of chapter 2, where:

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

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[8, 4, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Model__

As we've eluded to before, one can look at an image and immediately identify the centre of the galaxy. It's 
that bright blob of light in the middle! Given that we know we're going to make the model more complex in the 
next search, lets take a more liberal approach than before and fix the centre of the bulge and 
disk to $(y,x)$ = (0.0", 0.0")..

Now, you might be thinking, doesn`t this prevent our search from generalizing to other galaxies? What if the 
centre of their galaxy isn't at (0.0", 0.0")?

Well, this is true if our dataset reduction centres the galaxy somewhere else. But we get to choose where we 
centre it when we make the image. Therefore, I`d recommend you always centre the galaxy at the same location, 
and (0.0", 0.0") seems the best choice!
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Exponential)

"""
You haven't actually seen a line like this one before. By setting a parameter to a number (and not a prior) it is be 
removed from non-linear parameter space and always fixed to that value. Pretty neat, huh?
"""
bulge.centre_0 = 0.0
bulge.centre_1 = 0.0
disk.centre_0 = 0.0
disk.centre_1 = 0.0

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model_1 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis__

Now lets create the search and analysis.
"""
search_1 = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_3"),
    name="tutorial_1_search_chaining_1",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = ag.AnalysisImaging(dataset=dataset, use_jax=True)

"""
Lets run the search, noting that our liberal approach to reducing the model complexity has reduced it to just 
6 parameters.
"""
print(
    "The non-linear search has begun running - checkout the workspace/output/5_chaining_searches"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

print("Search has finished run - you may now continue the notebook.")

"""
__Result__

The results are summarised in the `info` attribute.
"""
print(result_1.info)

"""
And indeed, we get a reasonably good model and fit to the data, in a much shorter space of time!
"""
fit_plotter = aplt.FitImagingPlotter(fit=result_1.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Prior Passing__

Now all we need to do is look at the results of search 1 and pass the results as priors for search 2. Lets setup 
a custom search that does exactly that.

`TruncatedGaussianPrior`'s are a nice way to pass priors. They tell the non-linear search where to look, but leave open the 
possibility that there might be a better solution nearby. In contrast, `UniformPrior`'s put hard limits on what values a 
parameter can or can`t take. It makes it more likely we will accidentally cut-out the global maxima solution.

Note that below the `disk` has become an `Sersic`.
"""
bulge = af.Model(ag.lp_linear.Sersic)
disk = af.Model(ag.lp_linear.Sersic)

"""
What I've done below is looked at the results of search 1 and manually specified a prior for every parameter. If a 
parameter was fixed in the previous search, its prior is based around the previous value. Don't worry about the sigma 
values for now, I've chosen values that I know will ensure reasonable sampling, but we'll cover this later.

__LENS BULGE PRIORS:__
"""
bulge.centre.centre_0 = af.TruncatedGaussianPrior(
    mean=0.0, sigma=0.1, lower_limit=-np.inf, upper_limit=np.inf
)
bulge.centre.centre_1 = af.TruncatedGaussianPrior(
    mean=0.0, sigma=0.1, lower_limit=-np.inf, upper_limit=np.inf
)
bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
bulge.effective_radius = af.TruncatedGaussianPrior(
    mean=0.75, sigma=0.4, lower_limit=0.0, upper_limit=np.inf
)
bulge.sersic_index = af.TruncatedGaussianPrior(
    mean=4.0, sigma=2.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__LENS DISK PRIORS:__
"""
disk.centre.centre_0 = af.TruncatedGaussianPrior(
    mean=0.0, sigma=0.1, lower_limit=-np.inf, upper_limit=np.inf
)
disk.centre.centre_1 = af.TruncatedGaussianPrior(
    mean=0.0, sigma=0.1, lower_limit=-np.inf, upper_limit=np.inf
)
disk.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
    mean=0.11, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
disk.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
    mean=0.05, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
disk.effective_radius = af.TruncatedGaussianPrior(
    mean=1.52, sigma=0.4, lower_limit=0.0, upper_limit=np.inf
)
disk.sersic_index = af.TruncatedGaussianPrior(
    mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=np.inf
)

"""
We now compose the model with these components that have had their priors customized. 
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model_2 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
The `info` attribute shows the model, including the priors specified above.
"""
print(model_2.info)

"""
Lets setup and run the search. As expected, it gives us the correct model. However, it does so significantly 
faster than we are used to!
"""
search_2 = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_3"),
    name="tutorial_1_search_chaining_2",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = ag.AnalysisImaging(dataset=dataset, use_jax=True)

print(
    "The non-linear search has begun running - checkout the workspace/output/5_chaining_searches"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

print("Search has finished run - you may now continue the notebook.")

"""
__Result__

We can again inspect the results via the `info` attribute.
"""
print(result_2.info)

"""
And a plot of the image shows we get a good model again!
"""
fit_plotter = aplt.FitImagingPlotter(fit=result_2.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Wrap Up__

Chaining two searches together was a huge success. We managed to fit a complex and realistic model, but were able to 
begin by making simplifying assumptions that eased our search of non-linear parameter space. We could apply search 1 to 
pretty much any galaxy and therefore get ourselves a decent model with which to tune search 2`s priors.

You are probably thinking though that there is one huge, giant, glaring flaw in all of this that I've not mentioned. 
Search 2 can`t be generalized to another lens, because its priors are tuned to the image we fitted. If we had a lot 
of galaxies, we`d have to write a new search for every single one. This isn't ideal, is it?

Fortunately, we can pass priors in **PyAutoGalaxy** without specifying the specific values. The API for this technique,
called prior passing, is the topic of the next tutorial.
"""
