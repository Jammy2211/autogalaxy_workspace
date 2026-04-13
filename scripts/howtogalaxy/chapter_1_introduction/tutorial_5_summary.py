"""
Tutorial 9: Summary
===================

In this chapter, we have learnt that:

 1) **PyAutoGalaxy** uses Cartesian `Grid2D`'s of $(y,x)$ coordinates to evaluate galaxy luminous emission.
 2) These grids are combined with light profiles to compute images and other quantities.
 3) Profiles are grouped together to make galaxies.
 4) Collections of galaxies (at the same redshift) can be made..
 5) The Universe's cosmology can be input into this `Galaxies` to convert its units to kiloparsecs.
 6) The galaxies's image can be used to simulate galaxy `Imaging` like it was observed with a real telescope.
 7) This data can be fitted, so to as quantify how well a model galaxy system represents the observed image.

In this summary, we'll go over all the different Python objects introduced throughout this chapter and consider how
they come together as one.

__Contents__

**Initial Setup:** Create profiles, galaxies and a Galaxies object for illustration.
**Object Composition:** How Galaxies, Galaxy and Profile objects compose together.
**Visualization:** Customize and visualize any aspect of galaxies using the plotting API.
**Code Design:** Discussion of PyAutoGalaxy's object-oriented design philosophy.
**Source Code:** Links to the source code repositories for PyAutoFit, PyAutoArray and PyAutoGalaxy.
**Wrap Up:** Summary of chapter 1 and preview of the modeling chapter.
"""

# from autoconf import setup_notebook; setup_notebook()

from pathlib import Path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Initial Setup__

Below, we do all the steps we have learned this chapter, making profiles, galaxies, etc. 

Note that we use two galaxies, the first of which has a bulge and disk.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

galaxy_0 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
    disk=ag.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
    ),
)

galaxy_1 = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(1.0, 1.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

"""
__Object Composition__

Lets now consider how all of the objects we've covered throughout this chapter (`LightProfile`'s, `MassProfile`'s,
`Galaxy`'s, `Galaxies`'s) come together.

The `Galaxies` contain the `Galaxy`'s which contains the `Profile`'s:
"""
print(galaxies[0])
print()
print(galaxies[0].bulge)
print()
print(galaxies[0].disk)
print()
print(galaxies[1].bulge)
print()

"""
Once we have defined the galaxies, we can plot any quantity introduced throughout this chapter for a specific component, 
a single galaxy, or multiple galaxies as needed.

For example, if we want to plot the image of the first galaxy's bulge and disk, we can do this in a variety of 
different ways.
"""
aplt.plot_array(array=galaxies.image_2d_from(grid=grid), title="Image")

aplt.plot_array(array=galaxies[0].image_2d_from(grid=grid), title="Image")

"""
Understanding how these objects decompose into the different components of a galaxy is important for general 
**PyAutoGalaxy** use.

As the galaxy systems that we analyse become more complex, it is useful to know how to decompose their light 
profiles, galaxies and galaxies to extract different pieces of information about the galaxy. 

For example, we made our galaxy above with two light profiles, a `bulge` and `disk`. We can plot the image of 
each component individually, now that we know how to break-up the different components of the galaxies.
"""
aplt.plot_array(array=galaxies[0].bulge.image_2d_from(grid=grid), title="Bulge Image")

aplt.plot_array(array=galaxies[0].disk.image_2d_from(grid=grid), title="Disk Image")

"""
__Visualization__

Furthermore, using the `MatPLot2D` and `Visuals2D` objects we can visualize any aspect we're interested 
in and fully customize the figure. 

Before beginning chapter 2 of **HowToGalaxy**, you should checkout the package `autogalaxy_workspace/plot`. 
This provides a full API reference of every plotting option in **PyAutoGalaxy**, allowing you to create your own 
fully customized figures of galaxies with minimal effort!
"""
aplt.plot_array(array=galaxies[0].bulge.image_2d_from(grid=grid), title="Bulge Image")

"""
And, we're done, not just with the tutorial, but the chapter!

__Code Design__

To end, I want to quickly talk about the **PyAutoGalaxy** code-design and structure, which was really the main topic of
this tutoriag.

Throughout this chapter, we never talk about anything like it was code. We didn`t refer to  'variables', 'parameters`' 
'functions' or 'dictionaries', did we? Instead, we talked about 'galaxies'. We discussed 
the objects that we, as scientists, think about when we consider a galaxy system.

Software that abstracts the underlying code in this way follows an `object-oriented design`, and it is our hope 
with **PyAutoGalaxy** that we've made its interface (often called the API for short) very intuitive, whether you were
previous familiar with galaxy morphology or a complete newcomer!

__Source Code__

If you do enjoy code, variables, functions, and parameters, you may want to dig deeper into the **PyAutoGalaxy** source 
code at some point in the future. Firstly, you should note that all of the code we discuss throughout the **HowToGalaxy** 
lectures is not contained in just one project (e.g. the **PyAutoGalaxy** GitHub repository) but in fact three repositories:

**PyAutoFit** - Everything required for modeling (the topic of chapter 2): https://github.com/rhayes777/PyAutoFit

**PyAutoArray** - Handles all data structures and Astronomy dataset objects: https://github.com/Jammy2211/PyAutoArray

**PyAutoGalaxy** - Contains the light profiles and galaxies: https://github.com/Jammy2211/PyAutoGalaxy

Instructions on how to build these projects from source are provided here:

https://pyautogalaxy.readthedocs.io/en/latest/installation/source.html

We take a lot of pride in our source code, so I can promise you its well written, well documented and thoroughly 
tested (check out the `test` directory if you're curious how to test code well!).

__Wrap Up__

You`ve learn a lot in this chapter, but what you have not learnt is how to 'model' a real galaxy.

In the real world, we have no idea what the 'correct' combination of light profiles are that will give a good fit to 
a galaxy. Modeling is the process of finding the model which provides a good fit and it is the topic of chapter 2 
of **HowToGalaxy**.

Finally, if you enjoyed doing the **HowToGalaxy** tutorials please git us a star on the **PyAutoGalaxy** GitHub
repository: 

 https://github.com/Jammy2211/PyAutoGalaxy

Even the smallest bit of exposure via a GitHub star can help our project grow!
"""
