The ``data_preparation/imaging`` package provides tools for preparing an imaging
dataset (e.g. Hubble Space Telescope) before **PyAutoGalaxy** analysis:

Files (Beginner)
----------------

- ``casa_to_autogalaxy``: Convert a dataset to **PyAutoGalaxy** formats via CASA.
- ``profiling.py``: An overview of interferometer run-times and how to make your analysis run as fast as possible.

Files (Beginner / In Imaging)
-----------------------------

The following scripts are used to prepare components of an interferometer dataset, however they are used in an
identical fashion for dataset datasets.

Therefore, they are not located in the `interferometer/data_preparation` package, but instead in the
`data_preparation/imaging` package, so refer there for a description of their usage.

Note that in order to perform some tasks (e.g. mark on the image where the source is), you will need to use an image
of the interferometer data even though visibilities are used for the analysis.

- ``positions.py``: Marking source-positions on a lensed source such that mass models during a non-linear search are discarded if they do not trace close to one another.
- ``light_centre.py``: Masking the centre of the galaxy(s) light to help compose the model.
- ``extra_galaxies_centres.py``: Adding additional extra galaxy centres, which add extra light and mass profiles to a model.
- ``info.py``: Adding information to the dataset (e.g. redshifts) to aid analysis after modeling.