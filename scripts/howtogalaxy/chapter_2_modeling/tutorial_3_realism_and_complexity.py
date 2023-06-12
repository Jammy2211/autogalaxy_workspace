"""
Tutorial 3: Realism and Complexity
==================================

In the previous two tutorials, we fitted a fairly basic model: the galaxy's light was a single bulge component.
In real observations we know that galaxies are observed to have multiple different morphological structures.

In this tutorial, we'll use a more realistic model, which consists of the following light profiles:

 - An `Sersic` light profile for the galaxy's bulge [7 parameters].
 - An `Exponential` light profile for the galaxy's disk [6 parameters]

This model has 13 free parameters, meaning that the parameter space and likelihood function it defines has a
dimensionality of N=13. This is over double the number of parameters and dimensions of the models we fitted in the
previous tutorials and in future exercises, we will fit even more complex models with some 13+ parameters.

Therefore, take note, as we make our model more realistic, we also make its parameter space more complex, this is
an important concept to keep in mind for the remainder of this chapter!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt
import autofit as af

"""
__Initial Setup__

we'll use new galaxying data, where:

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

We'll create and use a 2.5" `Mask2D`, which is slightly smaller than the masks we used in previous tutorials.
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
__Model + Search + Analysis__

Now lets fit the dataset using a search.
"""
model = af.Collection(
    galaxies=af.Collection(
        galaxy=af.Model(
            ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Exponential
        )
    )
)

search = af.DynestyStatic(
    path_prefix=path.join("howtogalaxy", "chapter_2"),
    name="tutorial_3_realism_and_complexity",
    unique_tag=dataset_name,
    nlive=80,
    number_of_cores=1,
)

analysis = ag.AnalysisImaging(dataset=dataset)

print(
    "Dynesty has begun running - checkout the autogalaxy_workspace/output/howtogalaxy/chapter_2/tutorial_3_realism_and_complexity"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("Dynesty has finished run - you may now continue the notebook.")

"""
__Result__

Inspection of the `info` summary of the result suggests the model has gone to reasonable values.
"""
print(result.info)

"""
And lets look at how well the model fits the imaging data, which as we are used to fits the data brilliantly!
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Global and Local Maxima__

Up to now, all of our non-linear searches have been successes. They find a model that provides a visibly good fit
to the data, minimizing the residuals and inferring a high log likelihood value. 

These solutions are called 'global maxima', they correspond to the highest likelihood regions over all of parameter 
space. There are no other models in parameter space that would give higher likelihoods, this is the model we want 
to always infer!

However, non-linear searches may not always successfully locate the global maxima models. They may instead infer 
a 'local maxima', a solution which has a high log likelihood value relative to the models near it in parameter 
space, but where the log likelihood is significantly below the global maxima solution located somewhere else in 
parameter space. 

Why does a non-linear search infer these local maxima solutions? As discussed previously, the search guesses many 
models over and over, guessing more models in regions of parameter space where previous guesses gave the highest 
likelihood solutions. The search gradually 'converges' around any solution that gives a higher likelihood than the 
models nearby it in parameter space. If the search is not thorough enough, it may converge around a solution that 
appears to give a high likelihood (compared to the models around it) but, as discussed, is only a local maxima over 
all of parameter space.

Inferring such solutions is essentially a failure of our non-linear search and it is something we do not want to
happen! Lets infer a local maxima, by reducing the number of live points, `nlive`, dynesty uses to map out 
parameter space. We are going to use so few that the initial search over parameter space has an extremely low 
probability of getting close the global maxima, meaning it converges on a local maxima. 
"""
search = af.DynestyStatic(
    path_prefix=path.join("howtogalaxy", "chapter_2"),
    name="tutorial_3_realism_and_complexity__local_maxima",
    unique_tag=dataset_name,
    nlive=20,
    number_of_cores=1,
)

print(
    "Dynesty has begun running - checkout the autogalaxy_workspace/output/3_realism_and_complexity"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

result_local_maxima = search.fit(model=model, analysis=analysis)

print("Dynesty has finished run - you may now continue the notebook.")

"""
__Result__

Inspection of the `info` summary of the result suggests certain parameters have gone to different values to the fit
performed above.
"""
print(result_local_maxima.info)

"""
Lats look at the fit to the `Imaging` data, which is clearly worse than our original fit above.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result_local_maxima.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finally, just to be sure we hit a local maxima, lets compare the maximum log likelihood values of the two results 

The local maxima value is significantly lower, confirming that our non-linear search simply failed to locate lens 
models which fit the data better when it searched parameter space.
"""
print("Likelihood of Global Model:")
print(result.max_log_likelihood_fit.log_likelihood)
print("Likelihood of Local Model:")
print(result_local_maxima.max_log_likelihood_fit.log_likelihood)

"""
__Wrap Up__

In this example, we intentionally made our non-linear search fail, by using so few live points it had no hope of 
sampling parameter space thoroughly. For modeling real galaxies we wouldn't do this intentionally, but the risk of 
inferring a local maxima is still very real, especially as we make our model more complex.

Lets think about *complexity*. As we make our model more realistic, we also made it more complex. For this 
tutorial, our non-linear parameter space went from 7 dimensions to 13. This means there was a much larger *volume* of 
parameter space to search. As this volume grows, there becomes a higher chance that our non-linear search gets lost 
and infers a local maxima, especially if we don't set it up with enough live points!

At its core, modeling is all about learning how to get a non-linear search to find the global maxima region of 
parameter space, even when the model is complex. This will be the main theme throughout the rest of this chapter
and is the main subject of chapter 3.

In the next exercise, we'll learn how to deal with failure and begin thinking about how we can ensure our non-linear 
search finds the global-maximum log likelihood solution. First, think about the following:

 1) When you look at an image of a galaxy, do you get a sense of roughly what values certain model 
 parameters are?
    
 2) The non-linear search failed because parameter space was too complex. Could we make it less complex, whilst 
 still keeping our model fairly realistic?
    
 3) The galaxy in this example had only 7 non-linear parameters. Real galaxies may have multiple components (e.g. a 
 disk, bulge, bar, star-forming knot) and there may even be more than 1 galaxy! Do you think there is any hope of 
 us navigating a parameter space if the galaxies contributes 30+ parameters?
"""
