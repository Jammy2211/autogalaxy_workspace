"""
Tutorial 1: pixelizations
=========================

In the previous chapters, we used light profiles to model the light of a galaxy, where the light profile was an
analytic description of how the luminosity varies as a function of radius.

In this chapter, we are instead going to reconstruct the galaxy's light on a pixel-grid, and in this tutorial we will
learn how to create a pixelization in **PyAutoGalaxy**.

__Contents__

**Initial Setup:** Create a grid for illustration.
**Mesh:** Set up a rectangular mesh for the pixelization.
**Wrap Up:** Summary of pixelization concepts.
"""

# from autoconf import setup_notebook; setup_notebook()

import autogalaxy as ag
import autogalaxy.plot as aplt
from autoarray.inversion.plot.mapper_plots import plot_mapper

"""
__Initial Setup__

Lets setup a grid. 
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Mesh__

Next, lets set up a `Mesh` using the `mesh` module. The mesh represents the pixel-grid used by the pixelization
to reconstruct the galaxy.

There are multiple `Mesh`'s available in **PyAutoGalaxy**. For now, we'll keep it simple and use a uniform 
rectangular grid, whose `shape` defines its $(y,x)$ dimensions. We will make it the same shape as the 2D grid.
"""
mesh = ag.mesh.RectangularAdaptDensity(shape=(100, 100))

"""
We now pass the mesh to a `Pixelization`.
"""
pixelization = ag.Pixelization(mesh=mesh)

"""
By itself, a pixelization does not tell us much. It has no grid of $(y,x)$ coordinates, no image, and no information
about the galaxy we are fitting. 

This information comes when we use the pixelization to create up a `Mapper`, which we perform below using the grid 
that we created above.
"""
interpolator = mesh.interpolator_from(
    source_plane_data_grid=grid, source_plane_mesh_grid=None
)

mapper = ag.Mapper(interpolator=interpolator)

"""
This `Mapper` is a `RectangularMapper` -- every `Mesh` and `Pixelization` generates it owns mapper.
"""
print(type(mapper))

"""
By plotting our mapper, we now see our `Pixelization`. Its a fairly boring grid of rectangular pixels.
"""
plot_mapper(
    mapper=mapper, title="Fairly Boring Grid2D of RectangularAdaptDensity Pixels"
)

"""
However, the `Mapper` does contain lots of interesting information about our `Pixelization`, for example its 
pixelization_grid tells us where the pixel centers are located.
"""
print("RectangularAdaptDensity Grid2D Pixel Centre 1:")
print(mapper.source_plane_mesh_grid[0])
print("RectangularAdaptDensity Grid2D Pixel Centre 2:")
print(mapper.source_plane_mesh_grid[1])
print("RectangularAdaptDensity Grid2D Pixel Centre 3:")
print(mapper.source_plane_mesh_grid[2])
print("etc.")

"""
We can plot these centre on our grid, to make it look slightly less boring!
"""
plot_mapper(
    mapper=mapper,
    mesh_grid=mapper.source_plane_mesh_grid,
    title="Recntagular Grid With Pixel Cenres",
)

"""
The `Mapper` also has the grid that we passed when we set it up. Lets check they`re the same.
"""
print("Source Grid2D Pixel 1")
print(grid[0])
print(mapper.source_plane_data_grid[0])
print("Source Grid2D Pixel 2")
print(grid[1])
print(mapper.source_plane_data_grid[1])
print("etc.")

"""
We can over-lay this grid on the figure, which is starting to look a bit less boring now!
"""
plot_mapper(
    mapper=mapper,
    mesh_grid=mapper.source_plane_data_grid,
    title="Even less Boring Grid2D of RectangularAdaptDensity Pixels",
)

plot_mapper(
    mapper=mapper,
    mesh_grid=mapper.source_plane_data_grid,
    title="Zoomed Grid2D of RectangularAdaptDensity Pixels",
)

"""
Finally, the mapper`s `mesh_grid` has lots of information about the pixelization, for example, the arc-second 
size and dimensions.
"""
print(mapper.source_plane_mesh_grid.geometry.shape_native_scaled)
print(mapper.source_plane_mesh_grid.geometry.scaled_maxima)
print(mapper.source_plane_mesh_grid.geometry.scaled_minima)

"""
__Wrap Up__

This was a relatively gentle overview of pixelizations, but one that was hopefully easy to follow. Think about the 
following questions before moving on to the next tutorial:

 1) The rectangular pixelization`s edges are aligned with the most exterior coordinates of the source-grid. This is 
 intentional, why do you think this is?
"""
