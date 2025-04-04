"""
Results: CSV
============

In this tutorial, we use the aggregator to load the results of model-fits and output them in a single .csv file.

This enables the results of many model-fits to be concisely summarised and inspected in a single table, which
can also be easily passed on to other collaborators.

__Interferometer__

This script can easily be adapted to analyse the results of charge injection imaging model-fits.

The only entries that needs changing are:

 - `ImagingAgg` -> `InterferometerAgg`.
 - `FitImagingAgg` -> `FitInterferometerAgg`.
 - `ImagingPlotter` -> `InterferometerPlotter`.
 - `FitImagingPlotter` -> `FitInterferometerPlotter`.

Quantities specific to an interfometer, for example its uv-wavelengths real space mask, are accessed using the same API
(e.g. `values("dataset.uv_wavelengths")` and `.values{"dataset.real_space_mask")).

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is benefitial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
for i in range(2):

    dataset_name = f"simple"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = ag.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    bulge = af.Model(ag.lp_linear.Sersic)
    disk = af.Model(ag.lp_linear.Exponential)
    bulge.centre = disk.centre

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    search = af.Nautilus(
        path_prefix=path.join("results_folder_csv"),
        name="results",
        unique_tag=f"simple_{i}",
        n_live=100,
        number_of_cores=1,
    )

    class AnalysisLatent(ag.AnalysisImaging):

        def compute_latent_variables(self, instance):
            return {"example_latent": instance.galaxies.galaxy.bulge.sersic_index * 2.0}

    analysis = AnalysisLatent(dataset=dataset)

    result = search.fit(model=model, analysis=analysis)

"""
__Aggregator__

First, set up the aggregator as shown in `start_here.py`.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "results_folder_csv"),
)

"""
We next extract the `AggregateCSV` object, which has specific functions for outputting results in a CSV format.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

"""
__Adding CSV Columns_

We first make a simple .csv which contains two columns, corresponding to the inferred median PDF values for
the centre of the bulge of the galaxy.

To do this, we use the `add_column` method, which adds a column to the .csv file we write at the end. Every time
we call `add_column` we add a new column to the .csv file.

The `results_folder` contained three model-fits to three different datasets, meaning that each `add_column` call
will add three rows, corresponding to the three model-fits.

This adds the median PDF value of the parameter to the .csv file, we show how to add other values later in this script.
"""
agg_csv.add_column(argument="galaxies.galaxy.bulge.effective_radius")
agg_csv.add_column(argument="galaxies.galaxy.bulge.sersic_index")

"""
__Saving the CSV__

We can now output the results of all our model-fits to the .csv file, using the `save` method.

This will output in your current working directory (e.g. the `autogalaxy_workspace`) as a .csv file containing the 
median PDF values of the parameters, have a quick look now to see the format of the .csv file.
"""
agg_csv.save(path="csv_simple.csv")

"""
__Customizing CSV Headers__

The headers of the .csv file are by default the argument input above based on the model. 

However, we can customize these headers using the `name` input of the `add_column` method, for example making them
shorter or more readable.

We recreate the `agg_csv` first, so that we begin adding columns to a new .csv file.
"""
agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_column(
    argument="galaxies.galaxy.bulge.effective_radius",
    name="bulge_effective_radius",
)
agg_csv.add_column(
    argument="galaxies.galaxy.bulge.sersic_index",
    name="bulge_sersic_index",
)

agg_csv.save(path="csv_simple_custom_headers.csv")

"""
__Maximum Likelihood Values__

We can also output the maximum likelihood values of each parameter to the .csv file, using the `use_max_log_likelihood`
input.
"""
# agg_csv = af.AggregateCSV(aggregator=agg)
#
# agg_csv.add_column(
#     argument="galaxies.galaxy.bulge.effective_radius",
#     name="bulge_effective_radius_max_lh",
#     use_max_log_likelihood=True,
# )
#
# agg_csv.save(path="csv_simple_max_likelihood.csv")

"""
__Errors__

We can also output PDF values at a given sigma confidence of each parameter to the .csv file, using 
the `use_values_at_sigma` input and specifying the sigma confidence.

Below, we add the values at 3.0 sigma confidence to the .csv file, in order to compute the errors you would 
subtract the median value from these values.

The method below adds two columns to the .csv file, corresponding to the values at the lower and upper sigma values.
"""
# agg_csv = af.AggregateCSV(aggregator=agg)
#
# agg_csv.add_column(
#     argument="galaxies.galaxy.bulge.effective_radius",
#     name="bulge_effective_radius_sigma",
#     use_values_at_sigma=3.0,
# )
#
# agg_csv.save(path="csv_simple_errors.csv")

"""
__Column List__

We can add a list of values to the .csv file, provided the list is the same length as the number of model-fits
in the aggregator.

A useful example of this would be adding the name of every dataset to the .csv file in a column on the left,
which would allow you to know which dataset each row corresponds to.

To make this list, we use the `Aggregator` to loop over the `search` objects and extract their `unique_tag`'s, which 
when we fitted the model above used the dataset names. This API can also be used to extract the `name` or `path_prefix`
of the search and build an informative list for the names of the subplots.

We then pass this list to the `add_column` method, which will add a column to the .csv file.
"""
# unique_tag_list = [search.unique_tag for search in agg.values("search")]

# agg_csv = af.AggregateCSV(aggregator=agg)

# agg_csv.add_column(
#     argument=unique_tag_list,
#     name="dataset_name",
# )

# agg_csv.save(path="csv_simple_dataset_name.csv")


"""
__Latent Variables__

Latent variables are not free model parameters but can be derived from the model, and they are described fully in
?.

This example was run with a latent variable called `example_latent`, and below we show that this latent variable
can be added to the .csv file using the same API as above.
"""
# agg_csv = af.AggregateCSV(aggregator=agg)
#
# agg_csv.add_column(
#     argument="example_latent",
# )
#
# agg_csv.save(path="csv_example_latent.csv")

"""
__Computed Columns__

We can also add columns to the .csv file that are computed from the median PDF instance values of the model.

To do this, we write a function which is input into the `add_computed_column` method, where this function takes the
median PDF instance as input and returns the computed value.

Below, we add a trivial example of a computed column, where the value is twice the sersic index of the bulge.
"""
def sersic_index_x2_from(instance):

    return 2.0 * instance.galaxies.galaxy.bulge.sersic_index

agg_csv = af.AggregateCSV(aggregator=agg)

agg_csv.add_computed_column(
    name="bulge_sersic_index_x2_computed",
    compute=sersic_index_x2_from,
)

agg_csv.save(path="csv_computed_columns.csv")