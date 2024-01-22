"""
Tutorial 3: galaxies
====================

This tutorial introduces `Galaxy` objects, which:

 - Are composed from collections of the light profiles introduced in the previous tutorial.
 - Combine these profiles such that their properties (e.g. an image) are correctly calculated
 as the combination of these profiles.
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
__Planes__

We can also group galaxies into a `Plane` object, which is constructed from a list of galaxies.
"""
plane = ag.Plane(galaxies=[galaxy_with_light_profile, galaxy_with_3_light_profiles])

"""
The plane has agl the same methods we've seen for light profiles and galaxies.

For example, the `image_2d_from` method sums up the individual images of every galaxy in the plane.
"""
plane_image_2d = plane.image_2d_from(grid=grid)

"""
The `PlanePlotter` also allows similar plots to be made as the `GalaxyPlotter`.
"""
plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid)
plane_plotter.subplot_galaxy_images()

"""
The `Plane` object may seem a bit redundant, in this tutorial is pretty much acts as a list galaxies! 

For the first two chapters of the **HowToGalaxy** tutorials the plane object will indeed pretty much only be
used in this way. However, it has dedicated functionality for advanced **PyAutoGalaxy** features and theefore
has a purpose, which hopefully you'll be clear on in the future!

__Log10__

The previous tutorial discussed how the light distributions of galaxies are closer to a log10 distribution than a 
linear one and showed a convenience method to plot the image in log10 space.

When plotting multiple galaxies, plotting in log10 space makes it easier to see by how much the galaxy images
overlap and blend with one another. 
"""
mat_plot = aplt.MatPlot2D(use_log10=True)

plane_plotter = aplt.PlanePlotter(plane=plane, grid=grid, mat_plot_2d=mat_plot)
plane_plotter.figures_2d(image=True)

"""
__Wrap Up__

Tutorial 3 complete! 

We've learnt that by grouping light profiles into a galaxy and galaxies into a plane we can sum the contribution of 
each profile to  compute the galaxy's image (and other properties).
"""
