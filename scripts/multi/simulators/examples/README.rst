The ``simulators`` folder contains example scripts for simulating multiple imaging datasets (E.g. multi-wavelength
Hubble Space Telescope).

Notes
-----

The ``multi`` extends the ``imaging`` package and readers should refer to the ``imaging`` package for descriptions of
how to simulate a wider variety of galaxies.

Files (Advanced)
----------------

- ``interferometer.py``: A dataset which demonstrates modeling imaging and interferometer data simultaneously.
- ``same_wavelength.py``: A dataset with multiple images that are observed at the same wavelength.
- ``wavelength_dependence.py``: A dataset which is used to demonstrate fitting a model which depends on the wavelength of the observation following a linear `y = mx + c` relation.
- ``dataset_offsets.py``: A dataset where each image has a small spatial offset from one another (e.g. due to pointing errors).