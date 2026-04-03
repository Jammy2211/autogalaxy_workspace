"""
Plots: Plotters Pixelization
============================

This example illustrates the API for plotting pixelized source reconstructions using the new function-based
plotting API and the dedicated `InversionPlotter` and `MapperPlotter` objects for mesh-specific visualizations.

__Start Here Notebook__

You should refer to the `plots/start_here.ipynb` notebook first for a description of how plotters work and the default
behaviour of plotting visuals.

__Contents__

**Setup:** Set up all objects (e.g. grid, dataset, fit) used to illustrate plotting.
**Fit Imaging:** Plot the fit of a model to an imaging dataset that uses a pixelization.
**Inversion:** Plot the inversion object which performs the linear algebra reconstructing the source galaxy.
**Mapper:** Plot the mapper object which maps image-plane pixels to the source-plane pixelization.

__Setup__

To illustrate plotting, we require standard objects like a grid, dataset and fit.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

dataset_name = "sersic_x2"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = ag.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

pixelization = ag.Pixelization(
    mesh=ag.mesh.RectangularAdaptDensity(shape=(25, 25)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxy = ag.Galaxy(redshift=1.0, pixelization=pixelization)
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)

"""
__Fit Imaging__

The fit of a model using a pixelized source reconstruction is plotted using `aplt.subplot_fit_imaging`.

This subplot shows the data, model image, residuals and source-plane reconstruction on the mesh.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Reconstruction__

The reconstructed source galaxy on the mesh can be plotted using `aplt.plot_array`, passing the reconstruction
mapped back to the native image frame.
"""
inversion = fit.inversion

aplt.plot_array(
    array=fit.model_data, title="Reconstruction"
)

"""
__Inversion Plotter__

The `InversionPlotter` provides dedicated methods for plotting the properties of an inversion and its mapper.

We can extract an `InversionPlotter` from the fit and use it to plot the reconstruction and regularization weights.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion)
inversion_plotter.figures_2d(reconstructed_operated_data=True)

"""
To plot properties specific to a single pixelization (mapper), we use `figures_2d_of_pixelization`, specifying
the `pixelization_index` of the mapper we want to plot.
"""
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0,
    reconstructed_operated_data=True,
    reconstruction=True,
    reconstruction_noise_map=True,
    regularization_weights=True,
)

"""
The `InversionPlotter` can also produce a subplot of the mapper, showing the image-plane pixels mapped to
the source-plane pixelization alongside the source-plane reconstruction.
"""
aplt.subplot_of_mapper(mapper_index=0, inversion=inversion)

"""
__Mapper__

The `Mapper` maps image-plane pixels to the source-plane pixelization.

We extract the mapper from the inversion and pass it to a `MapperPlotter`.
"""
galaxies_to_inversion = ag.GalaxiesToInversion(
    galaxies=galaxies,
    dataset=dataset,
)

inversion = galaxies_to_inversion.inversion

mapper_galaxy_dict = galaxies_to_inversion.mapper_galaxy_dict

mapper = list(mapper_galaxy_dict)[0]

mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.figure_2d()

"""
The `Mapper` can also be plotted with a subplot showing the original image alongside the source-plane reconstruction.
"""
mapper_plotter = aplt.MapperPlotter(mapper=mapper)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
The indexes of `Mapper` plots can be highlighted to show how certain image pixels map to the source plane.
"""
visuals = aplt.Visuals2D(indexes=[0, 1, 2, 3, 4])

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Mesh Grids__

The image and source plane mesh grids, showing the centre of every source pixel, can be computed and plotted.
"""
image_plane_mesh_grid = mapper.mask.derive_grid.unmasked

visuals_2d = aplt.Visuals2D(mesh_grid=image_plane_mesh_grid)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, visuals_2d=visuals_2d)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__DelaunayDrawer / VoronoiDrawer__

We can customize the filling of Voronoi cells using the `VoronoiDrawer` object which wraps the
method `matplotlib.fill()`:

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill.html
"""
delaunay_drawer = aplt.DelaunayDrawer(edgecolor="b", linewidth=1.0, linestyle="--")
# voronoi_drawer = aplt.VoronoiDrawer(edgecolor="b", linewidth=1.0, linestyle="--")

mat_plot = aplt.MatPlot2D(delaunay_drawer=delaunay_drawer)

inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=mat_plot)

try:
    inversion_plotter.figures_2d_of_pixelization(
        pixelization_index=0, reconstruction=True
    )
    inversion_plotter.subplot_of_mapper(mapper_index=0)
except ImportError:
    print(
        "You have not installed the Voronoi natural neighbor interpolation package, see instructions at top of notebook."
    )

"""
Finish.
"""
