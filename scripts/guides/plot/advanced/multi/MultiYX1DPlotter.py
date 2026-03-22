"""
Plots: MultiYX1DPlotter
=========================

This example illustrates how to plot multiple 1D profiles on the same figure using `aplt.plot_yx` or
standard matplotlib.

It uses the specific example of plotting the 1D images of two `LightProfile` objects on the same figure.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
First, lets create two simple `LightProfile`'s which we'll plot the 1D images of on the same figure.
"""
light_0 = ag.lp.Sersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    effective_radius=1.0,
    sersic_index=2.0,
)

light_1 = ag.lp.Sersic(
    centre=(0.0, 0.0),
    intensity=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    effective_radius=2.0,
    sersic_index=2.0,
)

"""
We also need the 2D grid the `LightProfile`'s are evaluated on.
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__1D Projected Grids__

To compute 1D profiles we create a projected 2D radial grid aligned with each light profile's major axis.
"""
grid_projected_0 = grid.grid_2d_radial_projected_from(
    centre=light_0.centre, angle=light_0.angle()
)

grid_projected_1 = grid.grid_2d_radial_projected_from(
    centre=light_1.centre, angle=light_1.angle()
)

"""
__Compute 1D Images__

We evaluate the 1D image of each light profile using the projected grid.
"""
image_1d_0 = light_0.image_2d_from(grid=grid_projected_0)
image_1d_1 = light_1.image_2d_from(grid=grid_projected_1)

"""
__Plot with aplt.plot_yx__

The `aplt.plot_yx` function plots a 1D (y, x) curve.

To overlay both profiles on the same figure we use standard matplotlib directly, which gives full control.
"""
aplt.plot_yx(y=image_1d_0, x=grid_projected_0[:, 1], title="Light Profile 1D")

"""
__Plot Multiple Profiles with Matplotlib__

To plot multiple 1D profiles on the same figure, use matplotlib directly.
"""
plt.plot(grid_projected_0[:, 1], image_1d_0, label="Light Profile 0 (r_eff=1.0)")
plt.plot(grid_projected_1[:, 1], image_1d_1, label="Light Profile 1 (r_eff=2.0)")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Intensity")
plt.legend()
plt.title("Multi 1D Light Profile Plot")
plt.show()
plt.close()

"""
__Output__

To save the figure to disk, use `plt.savefig` before `plt.show`, or pass output arguments to `aplt.plot_yx`.
"""
aplt.plot_yx(
    y=image_1d_0,
    x=grid_projected_0[:, 1],
    title="Light Profile 1D",
)

"""
Finish.
"""
