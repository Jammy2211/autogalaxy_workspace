"""
Tutorial 1: Grids
=================

In this tutorial, we introduce two-dimensional grids of Cartesian $(y,x)$ coordinates, which represent the coordinates
of an observed data-set (e.g. imaging). In subsequent tutorials, we will use these grids to evaluate models of a
galaxy's luminous emission and structure.

Grids are defined in units of 'arc-seconds', if you are not familiar with this term it is the distance unit commonly
used by Astronomers. **PyAutoGalaxy** automatically converts all grids from units of pixels to arc-seconds, so you
should simply get used to seeing distances displayed in arc seconds.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Grids__

In **PyAutoGalaxy**, a `Grid2D` is a set of two-dimensional $(y,x)$ coordinates (in arc-seconds) that are used to 
evaluate the luminous emission of a galaxy.

The $(y,x)$ coordinates on the `Grid2D` are aligned with the image we analyze, such that each coordinate maps to the 
centre of each image-pixel. Lets make a `Grid2D` on a grid of 100 x 100 pixels, with a pixel scale (arcsecond-to-pixel 
conversion factor) of 0.05", giving us a 5" x 5" grid.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
First, lets plot this `Grid2D`, which shows that it is a fairly boring uniform `Grid2D` of dots.
"""
mat_plot = aplt.MatPlot2D(
    title=aplt.Title(label="Fairly Boring Uniform Grid2D Of Dots")
)

grid_plotter = aplt.Grid2DPlotter(grid=grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
We can print each coordinate of this `Grid2D`, revealing that it consists of a set of arc-second coordinates (where the 
spacing between each coordinate corresponds to the `pixel_scales` of 0.05" defined above).
"""
print("(y,x) pixel 0:")
print(grid.native[0, 0])
print("(y,x) pixel 1:")
print(grid.native[0, 1])
print("(y,x) pixel 2:")
print(grid.native[0, 2])
print("(y,x) pixel 100:")
print(grid.native[1, 0])
print("etc.")

"""
__Data Structure__

Above, you may have noted that we use the `native` attribute of the grid to print its $(y,x)$ coordinates. Every 
`Grid2D` object is accessible via two attributes, `native` and `slim`, which store the grid as NumPy ndarrays of two 
different shapes:
 
 - `native`: an ndarray of shape [total_y_image_pixels, total_x_image_pixels, 2], which is the native shape of the 
 2D grid and corresponds to the resolution of the image datasets we pair with a grid.
 
 - `slim`: an ndarray of shape [total_y_image_pixels*total_x_image_pixels, 2] which is a slimmed-down representation 
 the grid which collapses the inner two dimensions of the native ndarray to a single dimension.
"""
print("(y,x) pixel 0 (accessed via native):")
print(grid.native[0, 0])
print("(y,x) pixel 0 (accessed via slim 1D):")
print(grid.slim[0])

"""
Currently, it is unclear why there is a need for a `slim` representation of the grid (as the native representation 
contains all the information about the grid in a structure that is more representative of the grid itself). This will 
become apparent throughout the **HowToGalaxy** lectures, so for now don't worry about it! 

The shapes of the `Grid2D` in its `native` and `slim` formats are also available, confirming that this grid has a 
`native` resolution of (100 x 100) and a `slim` resolution of 10000 coordinates.
"""
print(grid.shape_native)
print(grid.shape_slim)

"""
Note that neither of the shapes above include the third index of the `Grid2D` which has dimensions 2 (corresponding to 
the y and x coordinates). This is accessible by using the standard numpy `shape` method on each grid.

This is worth noting, as we will introduce addition data structures throughout the tutorials which use the same
`native` and `slim` notation but may not include this final dimension of size 2. This means that the `shape_native`
and `shape_slim` attributes can be used to compare the shapes of different data structures in a common format.
"""
print(grid.native.shape)
print(grid.slim.shape)

"""
We can print the entire `Grid2D` in its `slim` or `native` form. 
"""
print(grid.native)
print(grid.slim)

"""
__Wrap Up__

Congratulations, you`ve completed your first **PyAutoGalaxy** tutorial! Before moving on to the next one, experiment 
with  **PyAutoGalaxy** by doing the following:

1) Change the pixel-scale of the `Grid2D`'s: what happens to the arc-second's grid of coordinates?
2) Change the resolution of the `Grid2D`'s: what happens to the arc-second's grid of coordinates?
"""

"""
Tutorial 2: Light Profiles
==========================

This tutorial introduces `LightProfile` objects, which represent analytic forms for the light distribution of galaxies.

By passing these objects 2D grids of $(y,x)$ coordinates we can create images from a light profile, which is therefore
a model of a galaxy's luminous emission.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Initial Setup__

We first setup a `Grid2D`, which uses the same resolution and arc-second to pixel conversion as the previous tutorial.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Light Profiles__

We now create a `LightProfile` using the `light_profile` module, which is access via `lp` for  conciseness. 

wWe'll use  the elliptical Sersic light profile (using the `Sersic` object), which is an analytic function used 
throughout studies of galaxy morphology to represent their light. 

You'll note that we use `Ell` to concisely describe that this profile is ellipticag. If you are unsure what 
the `ell_comps` are, I'll give a description of them at the end of the tutorial.
"""
sersic_light_profile = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

"""
By printing a `LightProfile` we can display its parameters.
"""
print(sersic_light_profile)

"""
__Images__

We next pass the `Grid2D` to the `sersic_light_profile`, to compute the intensity of the Sersic at every (y,x) 
coordinate on our two dimension grid. This uses the `image_2d_from` method, and you'll see throughout this tutorial 
that **PyAutoGalaxy** has numerous `_from` methods for computing quantities from a grid.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

"""
Much like the `Grid2D` objects discussed in the previous tutorial, this returns an `Array2D` object:
"""
print(type(image))

"""
Just like a grid, the `Array2D` object has both `native` and `slim` attributes:
"""
print("Intensity of pixel 0:")
print(image.native[0, 0])
print("Intensity of pixel 1:")
print(image.slim[1])

"""
For an `Array2D`, the dimensions of these attributes are as follows:

 - `native`: an ndarray of shape [total_y_image_pixels, total_x_image_pixels].

 - `slim`: an ndarray of shape [total_y_image_pixels*total_x_image_pixels].

The `native` and `slim` dimensions are therefore analogous to those of the `Grid2D` object, but without the final 
dimension of 2.
"""
print(image.shape_native)
print(image.shape_slim)

"""
We can use a `LightProfilePlotter` to plot the image of a light profile. We pass this plotter the light profile and
a grid, which are used to create the image that is plotted.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid
)
light_profile_plotter.figures_2d(image=True)

"""
__Over Sampling__

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated 
on a higher resolution grid than the image data to ensure the calculation is accurate. 

For a new user, the details of over-sampling are not important, therefore just be aware that all calculations use an
adaptive over sampling scheme which high accuracy across all use cases.

Once you are more experienced, you should read up on over-sampling in more detail via 
the `autogalaxy_workspace/*/guides/over_sampling.ipynb` notebook.

__Log10__

The light distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.
Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
mat_plot = aplt.MatPlot2D(use_log10=True)

light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid, mat_plot_2d=mat_plot
)
light_profile_plotter.figures_2d(image=True)

"""
__Wrap Up__

Congratulations, you`ve completed your second **PyAutoGalaxy** tutorial! 

Before moving on to the next one, experiment with  **PyAutoGalaxy** by doing the following:

1) Change the `LightProfile`'s effective radius and Sersic index - how does the image's appearance change?
2) Experiment with different `LightProfile`'s and `MassProfile`'s in the `light_profile` module. 

__Elliptical Components___

The `ell_comps` describe the ellipticity of light distribution. 

We can define a coordinate system where an ellipse is defined in terms of:

 - axis_ratio = semi-major axis / semi-minor axis = b/a
 - position angle, where angle is in degrees.

See https://en.wikipedia.org/wiki/Ellipse for a full description of elliptical coordinates.

The elliptical components are related to the axis-ratio and position angle as follows:

    fac = (1 - axis_ratio) / (1 + axis_ratio)

    elliptical_comp[0] = elliptical_comp_y = fac * np.sin(2 * angle)
    elliptical_comp[1] = elliptical_comp_x = fac * np.cos(2 * angle)

We can use the `convert` module to determine the elliptical components from an `axis_ratio` and `angle`,
noting that the position angle is defined counter-clockwise from the positive x-axis.
"""
ell_comps = ag.convert.ell_comps_from(axis_ratio=0.5, angle=45.0)

print(ell_comps)

"""
The reason light profiles use the elliptical components instead of an axis-ratio and position angle is because it 
improves the modeling process. What is modeling? You'll find out in chapter 2!
"""

"""
Tutorial 3: galaxies
====================

This tutorial introduces `Galaxy` objects, which:

 - Are composed from collections of the light profiles introduced in the previous tutorial.

 - Combine these profiles such that their properties (e.g. an image) are correctly calculated as the combination of
 these profiles.

 - Also have a redshift, which defines how far a galaxy is from Earth.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Initial Setup__

Lets use the same `Grid2D` as the previous tutorial.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

Lets make a galaxy with an elliptical Sersic `LightProfile`, by simply passing this profile to a `Galaxy` object.
"""
sersic_light_profile = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

galaxy_with_light_profile = ag.Galaxy(redshift=0.5, light=sersic_light_profile)

print(galaxy_with_light_profile)

"""
We have seen that we can pass a 2D grid to a light profile to compute its image via its `image_2d_from` method. We 
can do the exact same with a galaxy:
"""
galaxy_image_2d = galaxy_with_light_profile.image_2d_from(grid=grid)

print("intensity of `Grid2D` pixel 0:")
print(galaxy_image_2d.native[0, 0])
print("intensity of `Grid2D` pixel 1:")
print(galaxy_image_2d.native[0, 1])
print("intensity of `Grid2D` pixel 2:")
print(galaxy_image_2d.native[0, 2])
print("etc.")

"""
A `GalaxyPlotter` allows us to the plot the image, just like the `LightProfilePlotter` did for a light profile.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_light_profile, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
__Multiple Profiles__

We can pass galaxies as many light profiles as we like to a `Galaxy`, so lets create a galaxy with three light profiles.
"""
light_profile_1 = ag.lp.SersicSph(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.5
)

light_profile_2 = ag.lp.SersicSph(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=3.0
)

light_profile_3 = ag.lp.SersicSph(
    centre=(1.0, -1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

galaxy_with_3_light_profiles = ag.Galaxy(
    redshift=0.5,
    light_1=light_profile_1,
    light_2=light_profile_2,
    light_3=light_profile_3,
)

print(galaxy_with_3_light_profiles)

"""
If we plot the galaxy, we see 3 blobs of light!

(The image of multiple light profiles is simply the sum of the image of each individual light profile).
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_3_light_profiles, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
We can also plot each individual `LightProfile` using the plotter's `subplot_of_light_profiles` method.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
We can plot all light profiles in 1D, showing their decomposition of how they make up the overall galaxy.

Remember that 1D plots use grids aligned with each individual light profile centre, thus the 1D plot does not
show how these 3 galaxies are misaligned in 2D.
"""
galaxy_plotter.figures_1d_decomposed(image=True)

"""
__Multiple Galaxies__

We can also group galaxies into a `Galaxies` object, which is constructed from a list of galaxies.
"""
galaxies = ag.Galaxies(
    galaxies=[galaxy_with_light_profile, galaxy_with_3_light_profiles]
)

"""
The galaxies has the same methods we've seen for light profiles and individual galaxies.

For example, the `image_2d_from` method sums up the individual images of every galaxy.
"""
image = galaxies.image_2d_from(grid=grid)

"""
The `GalaxiesPlotter` shares the same API as the `LightProfilePlotter` and `GalaxyPlotter`.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
A subplot can be made of each individual galaxy image.
"""
galaxies_plotter.subplot_galaxy_images()

"""
__Log10__

The previous tutorial discussed how the light distributions of galaxies are closer to a log10 distribution than a 
linear one and showed a convenience method to plot the image in log10 space.

When plotting multiple galaxies, plotting in log10 space makes it easier to see by how much the galaxy images
overlap and blend with one another. 
"""
mat_plot = aplt.MatPlot2D(use_log10=True)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=galaxies, grid=grid, mat_plot_2d=mat_plot
)
galaxies_plotter.figures_2d(image=True)

"""
__Wrap Up__

Tutorial 3 complete! 

We've learnt that by grouping light profiles into a galaxy and galaxies together we can sum the contribution of 
each profile to  compute the galaxy's image (and other properties).
"""