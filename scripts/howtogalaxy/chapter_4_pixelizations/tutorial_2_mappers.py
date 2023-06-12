"""
Tutorial 2: Mappers
===================

In the previous tutorial, we used a pixelization to create made a `Mapper`. However, it was not clear what a `Mapper`
does, why it was called a mapper and whether it was mapping anything at all!

Therefore, in this tutorial, we'll cover mappers in more detail.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
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

"""
Now, lets set up our `Grid2D` (using the image above).
"""
grid = ag.Grid2D.uniform(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

"""
__Mappers__

We now setup a `Pixelization` and use it to create a `Mapper` via the plane`s source-plane grid, just like we did in
the previous tutorial.

We will make its pixelization resolution half that of the grid above.
"""
mesh = ag.mesh.Rectangular(
    shape=(dataset.shape_native[0] / 2, dataset.shape_native[1] / 2)
)

pixelization = ag.Pixelization(mesh=mesh)

mapper_grids = pixelization.mapper_grids_from(source_plane_data_grid=grid)

mapper = ag.Mapper(mapper_grids=mapper_grids, regularization=None)

"""
We now plot the `Mapper` alongside the image we used to generate the source-plane grid.

Using the `Visuals2D` object we are also going to highlight specific grid coordinates certain colors, such that we
can see how they map from the image grid to the pixelization grid. 
"""
visuals = aplt.Visuals2D(
    indexes=[range(250), [150, 250, 350, 450, 550, 650, 750, 850, 950, 1050]]
)
include = aplt.Include2D(mapper_source_plane_data_grid=False)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper, visuals_2d=visuals, include_2d=include
)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
Using a mapper, we can now make these mappings appear the other way round. That is, we can input a pixelization pixel
index (of our rectangular grid) and highlight how all of the image-pixels that it contains map to the image-plane. 

Lets map source pixel 313, the central source-pixel, to the image. We observe that for a given rectangular pixelization
pixel, there are four image pixels.
"""
visuals = aplt.Visuals2D(pix_indexes=[[312]])
mapper_plotter = aplt.MapperPlotter(
    mapper=mapper, visuals_2d=visuals, include_2d=include
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
Okay, so I think we can agree, mapper's map things! More specifically, they map pixelization pixels to multiple pixels 
in the observed image of a galaxy.

__Mask__

Finally, lets repeat the steps that we performed above, but now using a masked image. By applying a `Mask2D`, the 
mapper only maps image-pixels that are not removed by the mask. This removes the (many) image pixels at the edge of the 
image, where the galaxy is not present.

Lets just have a quick look at these edges pixels:

Lets use an circular `Mask2D`, which will capture the central galaxy light and clumps.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.0
)

dataset = dataset.apply_mask(mask=mask)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

"""
We can now use the masked grid to create a new `Mapper` (using the same rectangular 25 x 25 pixelization 
as before).
"""
mapper_grids = mesh.mapper_grids_from(source_plane_data_grid=dataset.grid)

mapper = ag.Mapper(mapper_grids=mapper_grids, regularization=None)

"""
Lets plot it.
"""
include = aplt.Include2D(mask=True, mapper_source_plane_data_grid=False)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, include_2d=include)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
First, We can see a red circle of dots in both the image and pixelization, showing where the edge of the mask
maps too in the pixelization.

Now lets show that when we plot pixelization pixel indexes, they still appear in the same place in the image.
"""
visuals = aplt.Visuals2D(pix_indexes=[[312], [314], [316], [318]])
mapper_plotter = aplt.MapperPlotter(
    mapper=mapper, visuals_2d=visuals, include_2d=include
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Wrap Up__

In this tutorial, we learnt about mappers, and we used them to understand how the image and pixelization map to one 
another. Your exercises are:
        
 1) Think about how this could help us actually model galaxies. We have said we're going to reconstruct our galaxies 
 on the pixel-grid. So, how does knowing how each pixel maps to the image actually help us? If you`ve not got 
 any bright ideas, then worry not, that exactly what we're going to cover in the next tutorial.
"""
