"""
Tutorial 4: Data
================

Up to now, all of the image's we've created have come from light profiles.

Other than the actual light that profile emits there are no other effects in the images of that light profile. This
contrasts real data of a galaxy, where there are lots of other effects in the imagimg (noise, diffraction due to the
telescope optics, etc.).

In this example, we use **PyAutoGalaxy** to simulate Hubble Space Telescope (HST) imaging of a galaxy. By simulate,
we mean that this image will contain these effects that are present in real data.
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

We'll need a 2D grid to make the galaxy image we'll ultimately simulate as if it was observed with HST.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

"""
Now, lets setup our galaxy and galaxies.
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

"""
Lets look at the galaxies's image, which is the image we'll be simulating.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
__Simulations__

To simulate an image, we need to model how the light is diffracted as it enters the telescope's optics. 

We do this using a two dimensional convolution, where a blurring kernel is used to mimic the effect of distraction. In
Astronomy, the kernel representing blurring in a telescope is called the 'Point-Spread Function' (PSF) and it is 
represented using a `Kernel2D` object, which in this example is a 2D Gaussian.
"""
psf = ag.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
The simulation does not use the image plotted above. Instead, we use a slightly different image which is padded 
with zeros around its edge, based on the shape of the PSF that we will convolve the image with. This ensures 
edge-effects do not degrade our simulation`s PSF convolution.
"""
normal_image_2d = galaxies.image_2d_from(grid=grid)
padded_image_2d = galaxies.padded_image_2d_from(
    grid=grid, psf_shape_2d=psf.shape_native
)

print(normal_image_2d.shape_native)
print(padded_image_2d.shape_native)

"""
To simulate imaging data we create a `SimulatorImaging` object, which represents all of the effects that occur when
imaging data is acquired in a telescope, including:

 1) Diffraction due to the telescope optics: this uses the Point Spread Function defined above.
 
 2) The Background Sky: this is background light from the Universe that is observed in addition to the galaxy's 
 light (the image that is returned has this background sky subtracted, so it simply acts as a source of Poisson noise).
 
 3) Poisson noise: The number of counts observed in each pixel of the HST imaging detector is a random process 
 described by Poisson count statistics. Thus, Poisson noise is included due to the background sky and galaxy emission.
 
We pass the galaxies and grid to the simulator to create the image of the galaxy and add the above effects to it.
"""
simulator = ag.SimulatorImaging(
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True
)

dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)

"""
By plotting the image, we can see it has been blurred due to the telescope optics and that noise has been added.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True)

"""
__Output__

We'll finally output these files to `.fits` files, which is the data storage format used by Astronomers to store
images. Pretty much all data from telescope like HST comes in `.fits` format, and **PyAutoGalaxy** has built-in tools 
for manipulating `.fits` files.

The `dataset_path` specifies where the data is output, this time in the directory 
`autogalaxy_workspace/dataset/imaging/howtogalaxy/`, which contains many example images of galaxy 
distributed with the`autogalaxy_workspace`.
"""
dataset_path = path.join("dataset", "imaging", "howtogalaxy")
print("Dataset Path: ", dataset_path)

"""
Finally, output our simulated data to hard-disk. In the next tutorial we'll load our simulated imaging data from 
these `.fits` files and begin to analyse them!
"""
dataset.output_to_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    overwrite=True,
)

"""
Finish.
"""