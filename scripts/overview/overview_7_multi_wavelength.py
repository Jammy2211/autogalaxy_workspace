"""
Overview: Multi-Wavelength
--------------------------

**PyAutoGalaxy** supports the analysis of multiple datasets simultaneously, including many CCD imaging datasets
observed at different wavebands (e.g. red, blue, green) and combining imaging and interferometer datasets.

This enables multi-wavelength galaxy modeling, where the color of the galaxies vary across the datasets.

Multi-wavelength galaxy modeling offers a number of advantages:

- It provides a wealth of additional information to fit the galaxy model, boosting the signal-to-noise of the
 observations.

- Instrument systematic effects, for example an uncertain PSF, will impact the model less because they vary across
 each dataset.

- It overcomes challenges associated with the deblending the emission of a galaxy with other galaxies in datasets where
 multiple galaxies near one another in the line-of-sight are observed.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt
from os import path
import numpy as np

"""
__Colors__

For multi-wavelength imaging datasets, we begin by defining the colors of the multi-wavelength images. 

For this overview we use only two colors, green (g-band) and red (r-band), but extending this to more datasets
is straight forward.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every dataset in our multi-wavelength observations can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Multi-wavelength imaging datasets do not use any new objects or class in **PyAutoGalaxy**.

We simply use lists of the classes we are now familiar with, for example the `Imaging` class.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "simple"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset_list = [
    ag.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

"""
Here is what our r-band and g-band observations of this galaxy system looks like.

Note how in the r-band the galaxy bulge is brighter than the disk, whereas in the g-band the disk is brighter. 

The different variation of the colors of the galaxy is a powerful tool for galaxy modeling as it provides a lot more
information on the galaxy's morphology.
"""
for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the image we fit the galaxy model to the data, which we 
define and use to set up the `Imaging` object that the galaxy model fits.

For multi-wavelength galaxy modeling, we use the same mask for every dataset whenever possible. This is not absolutely 
necessary, but provides a more reliable analysis.
"""
mask_list = [
    ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
    for dataset in dataset_list
]


dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Analysis__

We create a list of `AnalysisImaging` objects for every dataset.
"""
analysis_list = [ag.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
We now introduce the key new aspect to the **PyAutoGalaxy** multi-dataset API, which is critical to fitting multiple 
datasets simultaneously.

We sum the list of analysis objects to create an overall `CombinedAnalysis` object, which we can use to fit the 
multi-wavelength imaging data, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis objects (e.g. the fit to each separate waveband).

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis and therefore each dataset.
"""
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation is performed on a 
different CPU.
"""
analysis.n_cores = 1

"""
__Model__

We compose an initial galaxy model as per usuag.
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Sersic)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
However, there is a problem for multi-wavelength datasets. Should the light profiles of the galaxy's bulge and disk
have the same parameters for each wavelength image?

The answer is no. At different wavelengths, different stars appear brighter or fainter, meaning that the overall
appearance of the bulge and disk will change. 

We therefore allow specific light profile parameters to vary across wavelength and act as additional free
parameters in the fit to each image. 

We do this using the combined analysis object as follows:
"""
analysis = analysis.with_free_parameters(
    model.galaxies.galaxy.bulge.intensity, model.galaxies.galaxy.disk.intensity
)

"""
In this simple overview, this has added two additional free parameters to the model whereby:

 - The galaxy bulge's intensity is different in both multi-wavelength images.
 - The galaxy disk's intensity is different in both multi-wavelength images.
 
It is entirely plausible that more parameters should be free to vary across wavelength (e.g. the bulge and disk 
`effective_radius` or `sersic_index` parameters). 

This choice ultimately depends on the quality of data being fitted and intended science goag. Regardless, it is clear
how the above API can be extended to add any number of additional free parameters.

__Search + Model Fit__

Fitting the model uses the same API we introduced in previous overviews.
"""
search = af.Nautilus(path_prefix="overview", name="multiwavelength")

"""
The result object returned by this model-fit is a list of `Result` objects, because we used a combined analysis.
Each result corresponds to each analysis created above and is there the fit to each dataset at each wavelength.
"""
result_list = search.fit(model=model, analysis=analysis)

"""
Plotting each result's galaxies shows that the bulge and disk appear different in each result, owning to their 
different intensities.
"""
for result in result_list:
    galaxy_plotter = aplt.GalaxyPlotter(
        galaxy=result.max_log_likelihood_galaxies[0], grid=result.grid
    )
    galaxy_plotter.subplot_of_light_profiles(image=True)


"""
__Wavelength Dependence__

In the example above, a free `intensity` parameter is created for every multi-wavelength dataset. This would add 5+ 
free parameters to the model if we had 5+ datasets, quickly making a complex model parameterization.

We can instead parameterize the intensity of the galaxy light profiles as a user defined function of 
wavelength, for example following a relation `y = (m * x) + c` -> `intensity = (m * wavelength) + c`.

By using a linear relation `y = mx + c` the free parameters are `m` and `c`, which does not scale with the number
of datasets. For datasets with multi-wavelength images (e.g. 5 or more) this allows us to parameterize the variation 
of parameters across the datasets in a way that does not lead to a very complex parameter space.

Below, we show how one would do this for the `intensity` of a galaxy's bulge, give three wavelengths corresponding
to a dataset observed in the g, r and I bands.
"""
wavelength_list = [464, 658, 806]

analysis_list = []

bulge_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

for wavelength, imaging in zip(wavelength_list, dataset_list):
    bulge_intensity = (wavelength * bulge_m) + bulge_c

    analysis_list.append(
        ag.AnalysisImaging(dataset=dataset).with_model(
            model.replacing({model.galaxies.galaxy.bulge.intensity: bulge_intensity})
        )
    )

"""
__Same Wavelength Datasets__

The above API can fit multiple datasets which are observed at the same wavelength.

For example, this allows the analysis of images of a galaxy before they are combined to a single frame via the 
multidrizzling data reduction process to remove correlated noise in the data.

The pointing of each observation, and therefore centering of each dataset, may vary in an unknown way. This
can be folded into the model and fitted for as follows:

TODO : ADD CODE EXAMPLE.

__Interferometry and Imaging__

The above API can combine modeling of imaging and interferometer datasets 
(see `autogalaxy_workspace/*/multi/modeling/imaging_and_interferometer.ipynb` for an example script showing 
this in full).

Below are mock galaxy images of a system observed at a green wavelength (g-band) and with an interferometer at
sub millimeter wavelengths. 

The galaxy appears completely different in the g-band and at sub-millimeter wavelengths, allowing us to contrast
where a galaxy emits ultraviolet to where dust is heated.
"""
