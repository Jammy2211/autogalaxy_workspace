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

from pathlib import Path
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
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
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
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="search[1]",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis_1 = ag.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes. For certain calculations in the
pixelization, this information has to be passed in before the pixelization is performed. Below, we do this for 3
inputs:

- `total_linear_light_profiles`: The number of linear light profiles in the model. This is 0 because we are not
  fitting any linear light profiles to the data, primarily because the lens light is omitted.

- `total_mapper_pixels`: The number of source pixels in the rectangular pixelization mesh. This is required to set up 
  the arrays that perform the linear algebra of the pixelization.

- `source_pixel_zeroed_indices`: The indices of source pixels on its edge, which when the source is reconstructed 
  are forced to values of zero, a technique tests have shown are required to give accruate lens models.
"""
mesh_shape = (20, 20)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = ag.Preloads(
    mapper_indices=ag.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=ag.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the model fitted in search 2, where:

 - The galaxy's bulge is an `Sersic` [0 parameters: parameters fixed from search 1].
 
 - The galaxy's disk is an `Exponential` [0 parameters: parameters fixed from search 1].

 - The galaxy's clumps are reconstructed `RectangularAdaptDensity` mesh with resolution as free parameters [2 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

This search allows us to very efficiently set up the resolution of the mesh and regularization coefficient 
of the regularization scheme, before using these models to refit the galaxy mass model.
"""
pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=result_1.instance.galaxies.galaxy.bulge,
    disk=result_1.instance.galaxies.galaxy.disk,
    pixelization=pixelization,
)

model_2 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search_2 = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="search[2]",
    unique_tag=dataset_name,
    n_live=50,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis_2 = ag.AnalysisImaging(
    dataset=dataset,
    settings_inversion=ag.SettingsInversion(use_border_relocator=True),
    preloads=preloads,
    use_jax=True,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search (Search 3)__

We use the results of searches 1 and 2 to create the model fitted in search 3, where:

 - The galaxy's bulge is an `Sersic` [7 parameters: priors initialized from search 1].
 
 - The galaxy's disk is an `Exponential` [6 parameters: priors initialized from search 1].

 - The galaxy's light uses a `RectangularAdaptDensity` mesh[parameters fixed to results of search 2].

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

pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=ag.reg.GaussianKernel,
)

galaxy = af.Model(
    ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk, pixelization=pixelization
)

model_3 = af.Collection(galaxies=af.Collection(galaxy=galaxy))

search_3 = af.Nautilus(
    path_prefix=Path("howtogalaxy", "chapter_4"),
    name="search[3]",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
)

analysis_3 = ag.AnalysisImaging(dataset=dataset, preloads=preloads, use_jax=True)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Wrap Up__

And with that, we now have a pipeline to model galaxies using an inversion! 
"""
