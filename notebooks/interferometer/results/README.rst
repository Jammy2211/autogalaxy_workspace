The ``results`` folder contains example scripts for using the results of a **PyAutoGalaxy** model-fit.

Files (Beginner)
----------------

- ``start_here.py``: An overview of inspecting results from an individual model-fit.

Folders (Beginner)
------------------

- ``examples``: Result inspection and analysis for different aspects of the fit and model types.

Files (In Imaging)
------------------

The following results are available in the ``autolens_workspace/*/imaging/results`` folder, but are fully
application to interferometer data and can therefore have code copy and pasted from them.

- ``samples.py``: Non-linear search model results (parameter estimates, errors, etc.).
- ``tracer.py``:  ``Tracer`` modeling results (images, convergences, etc.).
- ``fits.py``:  Fitting results of a model (model-images, chi-squareds, likelihoods, etc.).
- ``galaxies.py``:  Inspecting individual galaxies, light profiles and mass profile in a model.
- ``units_and_cosmology.py``: Unit conversions and Cosmological quantities (converting to kiloparsecs, Einstein masses, etc.).
- ``linear``:  Analysing the results of fits using linear light profiles via an inversion.
- ``pixelization``:  Analysing the results of a pixelized source reconstruction via an inversion.