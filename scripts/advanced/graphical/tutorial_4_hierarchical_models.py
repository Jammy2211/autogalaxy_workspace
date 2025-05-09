"""
Tutorial 4: Hierarchical Models
===============================

In the previous tutorial, we fitted a graphical model with the aim of determining an estimate of a shared
parameter, the `sersic_index` of three galaxy models. We did this by fitting all datasets simultaneously.
When there are shared parameters we wish to estimate, this is a powerful and effective tool, but for many graphical
models things are not so straight forward.

A common extension to this problem is one where we expect that the shared parameter(s) of the model do not have exactly
the same value in every dataset. Instead, our expectation is that the parameter(s) are drawn from a common
parent distribution (e.g. a Gaussian distribution). It is the parameters of this parent distribution that we consider
shared across the dataset, and these are the parameters we ultimately wish to infer to understand the global
behaviour of our model.

This is called a hierarchical model, and we will fit such a model In this tutorial. We will again fit a dataset
comprising 3 strong galaxies. However, the `sersic_index` of each model is no longer the same in each
dataset -- they are instead drawn from a shared parent Gaussian distribution with `mean=4.0` and `sigma=1.0`. Using
a hierarchical model we will recover these input values of the parent distribution's `mean` and `sigma`, by fitting
the dataset of all 3 galaxies simultaneously.

__Sample Simulation__

The dataset fitted in this example script is simulated imaging data of a sample of 3 galaxies.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the
script `autogalaxy_workspace/scripts/simulators/imaging/samples/sersic.py`.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autofit as af
from os import path

"""
__Dataset__

For each galaxy dataset in our sample we set up the correct path and load it by iterating over a for loop. 

We are loading a different dataset to the previous tutorials, where the galaxies only have a single bulge component
which each have different Sersic indexes which are drawn from a parent Gaussian distribution with a mean value 
of 4.0 and sigma of 1.0.

This data is not automatically provided with the autogalaxy workspace, and must be first simulated by running the 
script `autogalaxy_workspace/scripts/simulators/imaging/samples/sersic.py`. 
"""
dataset_label = "samples"
dataset_type = "imaging"
dataset_sample_name = "simple__sersic"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_sample_name)

total_datasets = 3

dataset_list = []

for dataset_index in range(total_datasets):
    dataset_sample_path = path.join(dataset_path, f"dataset_{dataset_index}")

    dataset_list.append(
        ag.Imaging.from_fits(
            data_path=path.join(dataset_sample_path, "data.fits"),
            psf_path=path.join(dataset_sample_path, "psf.fits"),
            noise_map_path=path.join(dataset_sample_path, "noise_map.fits"),
            pixel_scales=0.1,
        )
    )

"""
__Mask__

We now mask each galaxy in our dataset, using the imaging list we created above.

We will assume a 3.0" mask for every galaxy in the dataset is appropriate.
"""
masked_imaging_list = []

for dataset in dataset_list:
    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = ag.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[8, 4, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    masked_imaging_list.append(dataset)

"""
__Paths__

The path the results of all model-fits are output:
"""
path_prefix = path.join("imaging", "graphical")

"""
__Analysis__

For each dataset we now create a corresponding `AnalysisImaging` class, as we are used to doing for `Imaging` data.
"""
analysis_list = []

for masked_dataset in masked_imaging_list:
    analysis = ag.AnalysisImaging(dataset=masked_dataset)

    analysis_list.append(analysis)

"""
__Model Individual Factors__

We first set up a model for each galaxy, with an `Sersic` bulge, which we will use to fit the hierarchical model.

This uses a nearly identical for loop to the previous tutorials, however a shared `sersic_index` is no longer used and 
each bulge `Sersic` is given its own prior for the `sersic_index`. 
"""
model_list = []

for dataset_index in range(total_datasets):
    bulge = af.Model(ag.lp_linear.Sersic)

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    model_list.append(model)

"""
__Analysis Factors__

Now we have our `Analysis` classes and model components, we can compose our `AnalysisFactor`'s.

These are composed in the same way as for the graphical model in the previous tutorial.
"""
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):
    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

    analysis_factor_list.append(analysis_factor)


"""
__Model__

We now compose the hierarchical model that we fit, using the individual model components created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `sersic_index` of each individual galaxy bulge is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""
hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.GaussianPrior(mean=4.0, sigma=4.0, lower_limit=0.0, upper_limit=100.0),
    sigma=af.GaussianPrior(mean=0.5, sigma=0.5, lower_limit=0.0, upper_limit=100.0),
)

"""
We now add each of the individual bulge `sersic_index` parameters to the `hierarchical_factor`.

This composes the hierarchical model whereby the individual `sersic_index` of every light model in our dataset is now 
assumed to be drawn from a shared parent distribution. It is the `mean` and `sigma` of this distribution we are hoping 
to estimate.
"""
for model in model_list:
    hierarchical_factor.add_drawn_variable(model.galaxies.galaxy.bulge.sersic_index)

"""
__Factor Graph__

We now create the factor graph for this model, using the list of `AnalysisFactor`'s and the hierarchical factor.

Note that in previous tutorials, when we created the `FactorGraphModel` we only passed the list of `AnalysisFactor`'s,
which contained the necessary information on the model create the factor graph that was fitted. The `AnalysisFactor`'s
were created before we composed the `HierachicalFactor` and we pass it separately when composing the factor graph.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, hierarchical_factor)

"""
The factor graph model `info` attribute shows that the hierarchical factor's parameters are included in the model.
"""
print(factor_graph.global_prior_model.info)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
nautilus = af.Nautilus(
    path_prefix=path.join("imaging", "graphical"),
    name="tutorial_4_hierarchical_models",
    n_live=150,
)

result = nautilus.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

The result's `info` attribute shows the result, including the hierarchical factor's parameters.
"""
print(result.info)

"""
We can now inspect the inferred value of hierarchical factor's mean and sigma.

We see that they are consistent with the input values of `mean=4.0` and `sigma=1.0`.
"""
samples = result.samples

mean = samples.median_pdf(as_instance=False)[-2]

u1_error = samples.values_at_upper_sigma(sigma=1.0)[-2]
l1_error = samples.values_at_lower_sigma(sigma=1.0)[-2]

u3_error = samples.values_at_upper_sigma(sigma=3.0)[-2]
l3_error = samples.values_at_lower_sigma(sigma=3.0)[-2]

print(
    "Inferred value of the mean of the parent hierarchical distribution for the bulge Sersic indexes: \n"
)
print(f"{mean} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{mean} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

scatter = samples.median_pdf(as_instance=False)[-2]

u1_error = samples.values_at_upper_sigma(sigma=1.0)[-1]
l1_error = samples.values_at_lower_sigma(sigma=1.0)[-1]

u3_error = samples.values_at_upper_sigma(sigma=3.0)[-1]
l3_error = samples.values_at_lower_sigma(sigma=3.0)[-1]

print(
    "Inferred value of the scatter (the sigma value of the Gassuain) of the parent hierarchical distribution for the bulge Sersic indexes: \n"
)
print(f"{scatter} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{scatter} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")

"""
__Comparison to one-by-one Fits__

We can compare the inferred values above to the values inferred in the 
tutorial `tutorial_optional_hierarchical_individuag.py`.

This fits the hierarchical model is a much simpler way -- fitting each dataset one-by-one and then fitted those
the parent Gaussian distribution to those results.

For the results below, inferred on my laptop, we can see that the correct mean and scatter of the parent Gaussian is 
inferred but the errors are much larger than the graphical model fit above.
"""
print(
    "Inferred value of the mean of the parent hierarchical distribution for one-by-one fits: \n"
)
print(
    "3.942165163710145 (2.9852072261494715 5.510653199603088) [1.0 sigma confidence intervals]"
)
print(
    "3.942165163710145 (0.15113736662637473 38.24375830218321) [3.0 sigma confidence intervals]"
)

print(
    "Inferred value of the scatter of the parent hierarchical distribution for one-by-one fits: \n"
)
print(
    "2.0268485435417767 (1.0204266541047813 5.509765412291949) [1.0 sigma confidence intervals]"
)
print(
    "2.0268485435417767 (0.49712736770633165 44.896662781502044) [3.0 sigma confidence intervals]"
)

"""
__Benefits of Graphical Model__

In the optional tutorial `tutorial_optional_hierarchical_individual` we compare the results inferred in this script
via a graphical model to a simpler approach which fits each dataset one-by-one and infers the hierarchical parent
distribution's parameters afterwards.

The graphical model provides a more accurate and precise estimate of the parent distribution's parameters. This is 
because the fit to each dataset informs the hierarchical distribution's parameters, which in turn improves
constraints on the other datasets. In a hierarchical fit, we describe this as "the datasets talking to one another". 

For example, by itself, dataset_0 may give weak constraints on the Sersic index spanning the range 10 -> 45 at 1 sigma 
confidence. Now, consider if simultaneously all of the other datasets provide strong constraints on the 
hierarchical's distribution's parameters, such that its `mean = 40 +- 5.0` and `sigma = 5.0 +- 2.0` at 1 sigma 
confidence. 

This will significantly change our inferred parameters for dataset 0, as the other datasets inform us
that solutions where the Sersic index is well below approximately 30 are less likely, because they are inconsistent with
the parent hierarchical distribution's parameters!

For complex graphical models with many hierarchical factors, this phenomena of the "datasets talking to one another" 
can be crucial in breaking degeneracies between parameters and maximally extracting information from extremely large
datasets.

__Wrap Up__

By composing and fitting hierarchical models in the graphical modeling framework we can fit for global trends
within large datasets. The tools applied in this tutorial and the previous tutorial can be easily extended to 
compose complex graphical models, with multiple shared parameters and hierarchical factors.

However, there is a clear challenge scaling the graphical modeling framework up in this way: model complexity. As the 
model becomes more complex, an inadequate sampling of parameter space will lead one to infer local maxima. Furthermore,
one will soon hit computational limits on how many datasets can feasibly be fitted simultaneously, both in terms of
CPU time and memory limitations. 

Therefore, the next tutorial introduces expectation propagation, a framework that inspects the factor graph of a 
graphical model and partitions the model-fit into many separate fits on each graph node. When a fit is complete, 
it passes the information learned about the model to neighboring nodes. 

Therefore, graphs comprising hundreds of model components (and tens of thousands of parameters) can be fitted as 
many bite-sized model fits, where the model fitted at each node consists of just tens of parameters. This makes 
graphical models scalable to largest datasets and most complex models!
"""
