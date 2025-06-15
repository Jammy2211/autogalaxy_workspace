"""
Tutorial 2: Mass Profiles
=========================

**PyAutoGalaxy** is a spin off of the project **PyAutoGalaxy**, software which models strong gravitational lens
systems.

It therefore has numerous mass profiles, which are used to perform lensing calculations. Currently, there is no
obvious use for these objects in **PyAutoGalaxy**, but if you are interested in performing stellar dynamics
they may be a good starting point to implementing this functionality (contact us directly on GitHub if you are
interested in this!

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
__Mass Profiles__

To perform ray-tracing, we require `MassProfile`'s, which are created via the `mass_profile_list` module and which is 
accessed via `mp` for conciseness. 

A `MassProfile` is an analytic function that describes the distribution of mass in a galaxy, and therefore 
can be used to derive its surface-density, gravitational potential and, most importantly, its deflection angles. In
gravitational lensing, the deflection angles describe how light is deflected by the `MassProfile` due to the curvature 
of space-time.

You'll note that we use `Sph` to concisely describe that this profile is sphericag.
"""
sis_mass_profile = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)

print(sis_mass_profile)

"""
__Deflection Angles__

We can again use a `from_grid_` method to compute the deflection angles of a mass profile from a grid. 

The deflection angles are returned as the arc-second deflections of the grid's $(y,x)$ Cartesian components. This again 
uses the `Grid2D``s object meaning that we can access the deflection angles via the `native` and `slim` attributes. 

(If you are still unclear what exactly a deflection angle means or how it will help us with gravitational lensing,
things should become a lot clearer in tutorial 4 of this chapter. For now, just look at the pretty pictures they make!).
"""
mass_profile_deflections_yx_2d = sis_mass_profile.deflections_yx_2d_from(grid=grid)

print("deflection-angles of `Grid2D` pixel 0:")
print(mass_profile_deflections_yx_2d.native[0, 0])
print("deflection-angles of `Grid2D` pixel 1:")
print(mass_profile_deflections_yx_2d.slim[1])
print()

"""
A `MassProfilePlotter` can plot the deflection angles.

(The black and white lines are called the `critical curve` and `caustic`. we'll cover what these are in a later tutorial.)
"""
mass_profile_plottter = aplt.MassProfilePlotter(
    mass_profile=sis_mass_profile, grid=grid
)
mass_profile_plottter.figures_2d(deflections_y=True, deflections_x=True)

"""
__Other Properties__

`MassProfile`'s have a range of other properties that are used for lensing calculations, a couple of which we've plotted 
images of below:

 - `convergence`: The surface mass density of the mass profile in dimensionless units.
 - `potential`: The gravitational of the mass profile again in convenient dimensionless units.
 - `agnification`: Describes how much brighter each image-pixel appears due to focusing of light rays.

Extracting `Array2D`'s of these quantities from **PyAutoGalaxy** is exactly the same as for the image and deflection 
angles above.
"""
mass_profile_convergence = sis_mass_profile.convergence_2d_from(grid=grid)

mass_profile_potential = sis_mass_profile.potential_2d_from(grid=grid)

mass_profile_magnification = sis_mass_profile.magnification_2d_from(grid=grid)

"""
Plotting them is equally straight forward.
"""
mass_profile_plottter.figures_2d(convergence=True, potential=True, magnification=True)
