"""
Tutorial 5: Model-Fit
=====================

In the previous tutorials we used an inversion to reconstruct a complex galaxy. However, from the perspective of
a scientific analysis, it is not clear how useful this was. When I fit a galaxy with light profiles, I learn about
its brightness (`intensity`), size (`effective_radius`), compactness (`sersic_index`), etc.

What did I learn about the galaxy I reconstructed? Not a lot, perhaps.

Inversions are most useful when combined with light profiles. For the complex galaxy we used throughout this tutorial,
we can fit it with light profiles to quantify the properties of its `bulge` and `disk` components, whilst
simultaneously fitting the clumps with the inversion so as to ensure they do not impact the fit.

To illustrate modeling using an inversion this tutorial therefore revisits the complex galaxy model-fit that we
performed in tutorial 4 of chapter 3. This time, as you have probably guessed, we will fit part of the complex galaxy
using an inversion.

We will use search chaining to do this, first fitting the main galaxy components with light profiles, thereby
initializing the bulge and disk components. In the later searches we will switch to an `Inversion`.
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

we'll use complex galaxy data, where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
 - The galaxy's has four star forming clumps which are `Sersic` profiles.
"""
dataset_name = "complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)


dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()


"""
__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 we fit a model where:

 - The galaxy's bulge is an `Sersic` with fixed centre [5 parameters].
 
 - The galaxy's disk is an `Exponential` with fixed centre [4 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.
"""
bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)

bulge.centre_0 = 0.0
bulge.centre_1 = 0.0
disk.centre_0 = 0.0
disk.centre_1 = 0.0

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model_1 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search_1 = af.Nautilus(
    path_prefix=path.join("howtogalaxy", "chapter_4"),
    name="search[1]_source[lp]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = ag.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the model fitted in search 2, where:

 - The galaxy's bulge is an `Sersic` [0 parameters: parameters fixed from search 1].
 
 - The galaxy's disk is an `Exponential` [0 parameters: parameters fixed from search 1].

 - The galaxy's clumps are reconstructed `Rectangular` mesh with resolution 40 x 40 [0 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=1.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the galaxy mass model.
"""
mesh = af.Model(ag.mesh.Rectangular, shape=(40, 40))

pixelization = af.Model(ag.Pixelization, mesh=mesh, regularization=ag.reg.Constant)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=result_1.instance.galaxies.galaxy.bulge,
    disk=result_1.instance.galaxies.galaxy.disk,
    pixelization=pixelization,
)

model_2 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search_2 = af.Nautilus(
    path_prefix=path.join("howtogalaxy", "chapter_4"),
    name="search[2]_source[inversion_initialization]",
    unique_tag=dataset_name,
    n_live=50,
)

analysis_2 = ag.AnalysisImaging(
    dataset=dataset, settings_pixelization=ag.SettingsPixelization(use_border=True)
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the model fitted in search 3, where:

 - The galaxy's bulge is an `Sersic` [7 parameters: priors initialized from search 1].
 
 - The galaxy's disk is an `Exponential` [6 parameters: priors initialized from search 1].

 - The galaxy's light uses a `Rectangular` mesh[parameters fixed to results of search 2].

 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 2]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.

This search allows us to refit the bulge and disk components with an inversion that takes care of the clumps.
"""
bulge = af.Model(ag.lp.Sersic)
bulge.ell_comps = result_1.model.galaxies.galaxy.bulge.ell_comps
bulge.intensity = result_1.model.galaxies.galaxy.bulge.intensity
bulge.effective_radius = result_1.model.galaxies.galaxy.bulge.effective_radius
bulge.sersic_index = result_1.model.galaxies.galaxy.bulge.sersic_index

disk = af.Model(ag.lp.Sersic)
disk.ell_comps = result_1.model.galaxies.galaxy.disk.ell_comps
disk.intensity = result_1.model.galaxies.galaxy.disk.intensity
disk.effective_radius = result_1.model.galaxies.galaxy.disk.effective_radius

mesh = af.Model(ag.mesh.Rectangular, shape=(40, 40))

pixelization = af.Model(ag.Pixelization, mesh=mesh, regularization=ag.reg.Constant)

galaxy = af.Model(
    ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk, pixelization=pixelization
)

model_3 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search_3 = af.Nautilus(
    path_prefix=path.join("howtogalaxy", "chapter_4"),
    name="search[3]_source[pix]",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_3 = ag.AnalysisImaging(dataset=dataset)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Wrap Up__

And with that, we now have a pipeline to model galaxies using an inversion! 
"""
