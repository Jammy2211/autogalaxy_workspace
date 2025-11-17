"""
Tutorial 4: Bayesian Regularization
===================================

So far, we have:

 - Used pixelizations and mappers to map pixelization pixels to image-pixels and visa versa.
 - Successfully used an inversion to reconstruct a galaxy.
 - Seen that this reconstruction provides a good fit of the observed image, providing a high likelihood solution.

The explanation of *how* an inversion works has so far been overly simplified. You'll have noted the regularization
inputs which we have not so far discussed. This will be the topic of this tutorial, and where inversions become more
conceptually challenging!
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Initial Setup__

we'll use the same complex galaxy data as the previous tutorial, where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - The galaxy's has four star forming clumps which are `Sersic` profiles.
"""
dataset_name = "complex"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
__Convenience Function__

we're going to perform a lot of fits using an `Inversion` this tutorial. This would create a lot of code, so to keep 
things tidy, I've setup this function which handles it all for us.
"""


def perform_fit_with_galaxy(dataset, galaxy):
    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
    )

    dataset = dataset.apply_mask(mask=mask)

    galaxies = ag.Galaxies(galaxies=[galaxy])

    return ag.FitImaging(dataset=dataset, galaxies=galaxies)


"""
__Pixelization__

Okay, so lets look at our fit from the previous tutorial in more detail.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularMagnification(shape=(50, 50)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

fit = perform_fit_with_galaxy(dataset=dataset, galaxy=galaxy)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

inversion_plotter = aplt.InversionPlotter(inversion=fit.inversion)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Regularization__

The galaxy reconstruction looks pretty good! 

However, the high quality of this solution was possible because I chose a `coefficient` for the regularization input of
1.0. If we reduce this `coefficient` to 0.01, the galaxy reconstruction goes *very* weird.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularMagnification(shape=(50, 50)),
    regularization=ag.reg.Constant(coefficient=0.01),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

no_regularization_fit = perform_fit_with_galaxy(dataset=dataset, galaxy=galaxy)

fit_plotter = aplt.FitImagingPlotter(fit=no_regularization_fit)
fit_plotter.subplot_fit()

inversion_plotter = aplt.InversionPlotter(inversion=no_regularization_fit.inversion)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
So, what is happening here? Why does reducing the `coefficient` do this to our reconstruction? First, we need
to understand what regularization actually does!

When the inversion reconstructs the galaxy, it does not *only* compute the set of pixelization pixel fluxes that 
best-fit the image. It also regularizes this solution, whereby it goes to every pixel on the rectangular grid 
and computes the different between the reconstructed flux values of every pixel with its 4 neighboring pixels. 
If the difference in flux is large the solution is penalized, reducing its log likelihood. You can think of this as 
us applying a 'smoothness prior' on the reconstructed galaxy's light.

This smoothing adds a 'penalty term' to the log likelihood of an inversion which is the summed difference between the 
reconstructed fluxes of every pixelization pixel pair multiplied by the `coefficient`. By setting the regularization 
coefficient to zero, we set this penalty term to zero, meaning that regularization is completely omitted.

Why do we need to regularize our solution? We just saw why, if we do not apply this smoothness prior to the galaxy 
reconstruction, we `over-fit` the image and reconstruct a noisy galaxy with lots of extraneous features. This is what 
the aliasing chequer-board effect is caused by. If the inversions's sole aim is to maximize the log likelihood, it can 
do this by fitting *everything* accurately, including the noise.

Over-fitting is why regularization is necessary. Solutions like this will completely ruin our attempts to model a 
galaxy. By smoothing our galaxy reconstruction we ensure it does not over fit noise in the image. 

So, what happens if we apply a high value for the regularization coefficient?
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularMagnification(shape=(50, 50)),
    regularization=ag.reg.Constant(coefficient=100.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

high_regularization_fit = perform_fit_with_galaxy(dataset=dataset, galaxy=galaxy)

fit_plotter = aplt.FitImagingPlotter(fit=high_regularization_fit)
fit_plotter.subplot_fit()

inversion_plotter = aplt.InversionPlotter(inversion=high_regularization_fit.inversion)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
The figure above shows that we completely remove over-fitting. However, we now fit the image data less poorly,
due to the much higher level of smoothing.

So, we now understand what regularization is and why it is necessary. There is one nagging question that remains, how 
do I choose the regularization coefficient value? We can not use the log likelihood, as decreasing the regularization
coefficient will always increase the log likelihood, because less smoothing allows the reconstruction to fit 
the data better.
"""
print("Likelihood Without Regularization:")
print(no_regularization_fit.log_likelihood_with_regularization)
print("Likelihood With Normal Regularization:")
print(fit.log_likelihood_with_regularization)
print("Likelihood With High Regularization:")
print(high_regularization_fit.log_likelihood_with_regularization)

"""
__Bayesian Evidence__

For inversions, we therefore need a different goodness-of-fit measure to choose the appropriate level of regularization. 

For this, we invoke the `Bayesian Evidence`, which quantifies the goodness of the fit as follows:

 - It requires that the residuals of the fit are consistent with Gaussian noise (which is the type of noise expected 
 in the imaging data). If this Gaussian pattern is not visible in the residuals, the noise must have been over-fitted
 by the inversion. The Bayesian evidence will therefore decrease. If the image is fitted poorly due to over smoothing, 
 the residuals will again not appear Gaussian either, again producing a decrease in the Bayesian evidence value.

 - There can be many solutions which fit the data to the noise level, without over-fitting. To determine the best 
 solutions from these solutions, the Bayesian evidence therefore also quantifies the complexity of the galaxy 
 reconstruction. If an inversion requires many pixels and a low level of regularization to achieve a good fit, the 
 Bayesian evidence will decrease. The evidence penalizes solutions which are complex, which, in a Bayesian sense, are 
 less probable (you may want to look up `Occam`s Razor`).

The Bayesian evidence therefore ensures we only invoke a more complex galaxy reconstruction when the data absolutely 
necessitates it.

Lets take a look at the Bayesian evidence of the fits that we performed above, which is accessible from a `FitImaging` 
object via the `log_evidence` property:
"""
print("Bayesian Evidence Without Regularization:")
print(no_regularization_fit.log_evidence)
print("Bayesian Evidence With Normal Regularization:")
print(fit.log_evidence)
print("Bayesian Evidence With High Regularization:")
print(high_regularization_fit.log_evidence)

"""
As expected, the solution that we could see `by-eye` was the best solution corresponds to the highest log evidence 
solution.

__Non-Linear and Linear__

Before we end, lets consider which aspects of an inversion are linear and which are non-linear.

The linear part of the inversion is the step that solves for the reconstruct pixelization pixel fluxes, including 
accounting for the smoothing via regularizaton. We do not have to perform a non-linear search to determine the pixel
fluxes or compute the Bayesian evidence discussed above.

However, determining the regularization `coefficient` that maximizes the Bayesian log evidence is a non-linear problem 
that requires a non-linear search. The Bayesian evidence also depends on the grid resolution, which means the 
pixel-grid's `shape` parameter may also now become dimensions of non linear parameter space (albeit it is common
practise for us to simply use the resolution of the image data, or a multiple of this). 

Nevertheless, these total only 3 non-linear parameters, far fewer than the 20+ that are required when modeling such a
complex galaxy using light profiles for every individual clump! 

Here are a few questions for you to think about.

 1) We maximize the log evidence by using simpler galaxy reconstructions. Therefore, decreasing the pixel-grid 
 size should provide a higher log_evidence, provided it still has sufficiently high resolution to fit the image well 
 (and provided that the regularization coefficient is set to an appropriate value). Can you increase the log evidence 
 from the value above by changing these parameters, I've set you up with a code to do so below.
"""
pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularMagnification(shape=(50, 50)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)

fit = perform_fit_with_galaxy(dataset=dataset, galaxy=galaxy)

print("Previous Bayesian Evidence:")
print(3988.0716851250163)
print("New Bayesian Evidence:")
print(fit.log_evidence)

fit_plotter = aplt.FitImagingPlotter(fit=fit)
fit_plotter.subplot_fit()

""" 
__Detailed Description__

Below, I provide a more detailed discussion of the Bayesian evidence. It is not paramount that you understand this to
use **PyAutoGalaxy**, but I recommend you give it a read to get an intuition for how the evidence works.

The Bayesian log evidence quantifies the following 3 aspects of a fit to galaxy imaging data:

1) *The quality of the image reconstruction:*  The galaxy reconstruction is a linear inversion which uses the observed
 values in the image-data to fit it and reconstruct the galaxy. It is in principle able to perfectly reconstruct the
 image regardless of the image’s noise or the accuracy of the model (e.g. at infinite resolution without
 regularization). The problem is therefore ‘ill-posed’ and this is why regularization is necessary.

 However, this raises the question of what constitutes a ‘good’ solution? The Bayesian evidence defines this by
 assuming that the image data consists of independent Gaussian noise in every image pixel. A ‘good’ solution is one
 whose chi-squared residuals are consistent with Gaussian noise, producing a reduced chi-squared near 1.0 .Solutions
 which give a reduced chi squared below 1 are penalized for being overly complex and fitting the image’s noise, whereas
 solutions with a reduced chi-squared above are penalized for not invoking a more complex galaxy model when the data it
 is necessary to fit the data bettter. In both circumstances, these penalties reduce the inferred Bayesian evidence!

2) *The complexity of the galaxy reconstruction:* The log evidence estimates the number of pixelization pixels that are used 
 to reconstruct the image, after accounting for their correlation with one another due to regularization. Solutions that
 require fewer correlated galaxy pixels increase the Bayesian evidence. Thus, simpler and less complex galaxy 
 reconstructions are favoured.

3) *The signal-to-noise (S/N) of the image that is fitted:* The Bayesian evidence favours models which fit higher S/N
 realizations of the observed data (where the S/N is determined using the image-pixel variances, e.g. the noise-map). Up 
 to now, all **PyAutoGalaxy** fits assumed fixed variances, meaning that this aspect of the Bayeisan evidence has no impact 
 on the inferred evidence values. 
   
 The premise is that whilst increasing the variances of image pixels lowers their S/N values and therefore also
 decreases the log evidence, doing so may produce a net increase in log evidence. This occurs when the chi-squared 
 values of the image pixels whose variances are increased were initially very high (e.g. they were fit poorly by the 
 model).

In summary, the log evidence is maximized for solutions which most accurately reconstruct the highest S/N realization of
the observed image, without over-fitting its noise and using the fewest correlated pixelization pixels. By employing 
this framework throughout, **PyAutoGalaxy** objectively determines the final model following the principles of Bayesian
analysis and Occam’s Razor.
"""
