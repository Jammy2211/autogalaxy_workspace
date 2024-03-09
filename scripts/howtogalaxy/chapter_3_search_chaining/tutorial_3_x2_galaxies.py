"""
Tutorial 3: Two Galaxies
========================

Up to now, all the images we've fitted had one galaxy. However, we saw in chapter 1 that our galaxies object can
consist of multiple galaxies which each contribute to the overall emission. Multi-galaxy systems are challenging to
model, because they add an extra 5-10 parameters to the non-linear search per galaxy and, more problematically, the
degeneracies between the parameters of the light profiles of the galaxies can be severe.

However, we can still break their analysis down using multiple searches and give ourselves a shot at getting a good
model. Here, we're going to fit a double galaxy system, fitting as much about each individual galaxy before
fitting them simultaneously.

Up to now, I've put a focus on an analysis being generag. The script we write in this example is going to be the
opposite, specific to the image we're modeling. Fitting multiple galaxies is really difficult and writing a
pipeline that we can generalize to many galaxies isn't currently possible.
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
__Initial Setup__

we'll use new galaxying data, where:

 - There are two galaxy's whose `LightProfile`'s are both `Sersic`'s.
"""
dataset_name = "sersic_x2"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

"""
__Mask__

We need to choose our mask for the analysis. Given the light of both galaxies is present in the image we'll need to 
include all their light in the image, so lets use a large circular mask. 

We'll use this mask in all three searches, however you could imagine customizing it on a per-search basis to speed up
the analysis.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=6.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()


"""
__Paths__

All four searches will use the same `path_prefix`, so we write it here to avoid repetition.
"""
path_prefix = path.join("howtogalaxy", "chapter_3", "tutorial_3_x2_galaxies")

"""
__Search Chaining Approach__

Looking at the image, there are two blobs of light corresponding to the two galaxies. 

So, how can we break the modeling up? As follows:

 1) Fit and subtract the light of the left galaxy individually.
 2) Fit and subtract the light of the right galaxy individually.
 3) Use these results to initialize a fit which fits both galaxy's simultaneously.

So, with this in mind, we'll perform an analysis using 3 searches:

 1) Fit the light of the galaxy on the left of the image, at coordinates (0.0", -1.0").
 2) Fit the light of the galaxy on the right of the image, at coordinates (0.0", 1.0").
 4) Fit all relevant parameters simultaneously, using priors from searches 1, and 2.

__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 we fit a model where:

 - The left galaxy's light is a parametric linear `DevVaucouleurs` bulge with fixed centre [3 parameters].

 - the right galaxy's light is omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

__Notes__

The `DevVaucouleurs` is an `Sersic` profile with `sersic_index=4`.

We fix the centre of its light to (0.0, -1.0), the pixel we know the left galaxy's light centre peaks.

We use linear light profiles througout this script, given that the model is quite complex and this helps
simplify it.
"""
left_galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.DevVaucouleurs)
left_galaxy.bulge.centre_0 = 0.0
left_galaxy.bulge.centre_1 = -1.0

model_1 = af.Collection(galaxies=af.Collection(left_galaxy=left_galaxy))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__
"""
analysis_1 = ag.AnalysisImaging(dataset=dataset)

search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__left_galaxy_light[bulge_linear]",
    unique_tag=dataset_name,
    n_live=75,
)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Model (Search 2)__

Search 2 we fit a model where:

 - The left galaxy's light is a parametric linear `DevVaucouleurs` bulge [0 parameters: fixed from search 1].

 - The right galaxy's light is a parametric linear `DevVaucouleurs` bulge with a fixed centre [3 parameters].

 - The galaxy's mass  galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

We fix the centre of the right lens's light to (0.0, 1.0), the pixel we know the right galaxy's light centre peaks.

We also pass the result of the `left_galaxy` from search ` as an `instance`, which should improve the fitting of the
right lens.
"""
right_galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.DevVaucouleurs)
right_galaxy.bulge.centre_0 = 0.0
right_galaxy.bulge.centre_1 = 1.0

model_2 = af.Collection(
    galaxies=af.Collection(
        left_galaxy=result_1.instance.galaxies.left_galaxy, right_galaxy=right_galaxy
    )
)

"""
The `info` attribute shows the model, including how all priors are updated via prior passing.
"""
print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 2)__
"""
analysis_2 = ag.AnalysisImaging(dataset=dataset)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__right_galaxy_light[bulge_linear]",
    unique_tag=dataset_name,
    n_live=75,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__
The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_2.info)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

Search 4 we fit a model where:

 - The left galaxy's light is a parametric linear `Sersic` bulge with centre fixed [4 parameters: priors initialized 
 from search 1].

 - The right galaxy's light is a parametric linear `Sersic` bulge with centre fixed [4 parameters: priors initialized 
 from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=8.

We can use a special prior passing method to do this, called `take_attributes`. This scans the `DevVaucouleurs`
passed to the `take_attributes` method for all parameters which have the same name as the `Sersic` model,
and if their names are the same it passes their prior as a `model` (like we did above). Thus, it will locate all 6
parameters in common between the two profiles (centre, ell_comps, intensity, effective_radius) and pass those,
leaving the `sersic_index`'s priors as the default values.

The `take_attributes` method is used in many examples of prior passing, when we pass a simpler parameterization of a
model to a more complex model. Another good example would be passing the result of a `IsothermalSph` to an
`Isothermal`.
"""
left_galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
left_galaxy.bulge.take_attributes(result_1.model.galaxies.left_galaxy.bulge)

right_galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
right_galaxy.bulge.take_attributes(result_2.model.galaxies.right_galaxy.bulge)

model_3 = af.Collection(
    galaxies=af.Collection(left_galaxy=left_galaxy, right_galaxy=right_galaxy)
)

analysis_3 = ag.AnalysisImaging(dataset=dataset)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]_light_x2[bulge_linear]",
    unique_tag=dataset_name,
    n_live=100,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Result (Search 3)__

The final results are summarized in the `info` attribute.
"""
print(result_3.info)

"""
__Wrap Up__

We have successfully fitted multiple galaxies, but fitting each one-by-one.
"""
