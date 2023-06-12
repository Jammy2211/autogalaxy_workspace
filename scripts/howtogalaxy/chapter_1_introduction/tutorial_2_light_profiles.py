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
__Wrap Up__

Congratulations, you`ve completed your second **PyAutoGalaxy** tutorial! 

Before moving on to the next one, experiment with  **PyAutoGalaxy** by doing the following:

1) Change the `LightProfile`'s effective radius and Sersic index - how does the image's appearance change?
2) Experiment with different `LightProfile`'s and `MassProfile`'s in the `light_profile` module. 

___Elliptical Components___

The `ell_comps` describe the ellipticity of light distribution. 

We can define a coordinate system where an ellipse is defined in terms of:

 - axis_ratio = semi-major axis / semi-minor axis = b/a
 - position angle, where angle is in degrees.

See https://en.wikipedia.org/wiki/Ellipse for a full description of elliptical coordinates.

The elliptical components are related to the axis-ratio and position angle as follows:

    fac = (1 - axis_ratio) / (1 + axis_ratio)
    
    elliptical_comp[0] = elliptical_comp_y = fac * np.sin(2 * angle)
    elliptical_comp[1] = elliptical_comp_x = fac * np.cos(2 * angle)

We can use the **PyAutoGalaxy** `convert` module to determine the elliptical components from an `axis_ratio` and `angle`,
noting that the position angle is defined counter-clockwise from the positive x-axis.
"""
ell_comps = ag.convert.ell_comps_from(axis_ratio=0.5, angle=45.0)

print(ell_comps)

"""
The reason light profiles use the elliptical components instead of an axis-ratio and position angle is because it 
improves the modeling process. What is modeling? You'll find out in chapter 2!
"""
