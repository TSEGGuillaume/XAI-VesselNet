XAI-VesselNet : Explainable AI for deep vessel segmentation
===============

# Table of content

* [Introduction](#Introduction)
* [Warnings](#Warnings)
* [Installation](#Installation)
* [Usage](#Usage)
    * [Compute a contribution map](#Compute-a-contribution-map)
    * [Analyse a contribution map](#Analyse-a-contribution-map)
    * [Arguments](#Arguments)
* [Method](#Method)
    * [Dataset](#Dataset)
    * [Models](#Models)
    * [Voreen](#Voreen)
    * [Integrated Gradients](#Integrated-Gradients)
* [References](#References)
* [License](License) 
* [TODO list](#TODO) (*for developers*)

# Introduction

Over the past decade, the use of deep learning has become increasingly widespread in the field of image processing.

Despite the increase in performance that this technology allows, it also has some drawbacks. One of them is that this kind of algorithms behave like black-boxes.
This is a problem for critical applications, especially for medical applications where understanding and traceability of diagnoses are the keys to reliability and trust.  


This repository contains a framework that aims to explain deep models used for vessel segmentation using [*integrated gradients*](https://arxiv.org/abs/1703.01365).

This work is a part of a PhD thesis, "[Détection faiblement supervisée de pathologies vasculaires](https://www.theses.fr/en/s307470#)" (Weakly supervised detection of vascular pathologies).


# Warnings

 * Linux operating systems are not supported yet. The following only stands for Windows OS.
 * This work is still in progress. The model and data used are hard-coded.


# Installation

1. Clone this repository in your workspace

2. Create the requiered environement to execute __XAI-VesselNet__
> conda env create -n `<environement_name>` --file env_XAI-VesselNet.yml

3. Set your __PYTHONPATH__ environement variable to allow Python to find __XAI-VesselNet__'s modules :<br>
NB : this will only affect the current user's environment.
> setx PYTHONPATH "$Env:PYTHONPATH;`<workspace_path>`\XAI-VesselNet\modules".

4. Replace path in the [__*./default.ini*__](./default.ini) configuration file to fit your workspace environement :

> [PATHS]<br>workspace = `<workspace_path>`<br>
weights_dir = `<workspace_path>`\resources\weights<br>
data_dir = `<workspace_path>`\resources\data<br>
result_dir = `<workspace_path>`\results

NB : Replace the `<environement_name>` and `<workspace_path>` aliases by your environement name and your own path.

# Usage

First, you should activate your freshly created environement using 
> conda activate -n `<environement_name>`

## Compute a contribution map

Compute a contribution map.

> python integrated_gradients.py [-h] [--node ID_NODE | --centerline ID_CENTERLINE | --position X Y Z] [--verbose]

Arguments : see [arguments](#Arguments) below.

## Analyse a contribution map
 
Analyse the attribution map associated with the specified node / centerline / image position.

> python attribution_analysis.py [-h] [--node ID_NODE | --centerline ID_CENTERLINE | --position X Y Z] [--verbose]

Arguments : see [arguments](#Arguments) below.

## Arguments :

|arg name|param|type|description|
|----------|---------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| -h | N/A | N/A | show the help message and exit the script|
| -n | ID_NODE | int | the ID of the node (from Voreen graph) to explain. The corresponding logit of the model will be explained|
| -c | ID_CENTERLINE | int | the ID of the centerline (from Voreen graph) to explain. The logit of the corresponding to the middle point of the centerline will be explained|
| -p | X Y Z | int int int | the image position X,Y,Z to explain. The logit of the corresponding position will be explain.|
| -v(vvvv) | N/A | N/A | the verbose level. The number of 'v' defines the level.|
___
| 'v' count | verbosity level |
| ----------- | ----------- |
| 1 | DEBUG |
| 2 | INFO |
| 3 | WARNING |
| 4 | ERROR |
| 5 | CRITICAL |

# Method

In the long term, this framework is intended to explain any type of model used in the vessel segmentation workflow.

On a trial basis, this program is intended to explain models for liver vessel segmentation.
This means that the model's weights and data in this repository are only for the application to liver vessels.

## Dataset

Data used in this framework comes from the modification of public [3D-IRCADb-01](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) dataset. The preprocessed version are available [here](http://eidolon.univ-lyon2.fr/~jlamy/).

The data used for models training are IRCADb patients [1-19] and the test data is the IRCADb patient 20.

## Models

Supported models are listed below :
* Dense U-Net
    * Trained from [Jerman data](./resources/weights/DenseUNet_Jerman_25625680.h5)

## Voreen

TODO

## Integrated Gradients

TODO

# References
* [Soler, L., A. Hostettler, V. Agnus, A. Charnoz, J. Fasquel, J. Moreau, A. Osswald, M. Bouhadjar, and J. Marescaux. “3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database.” IRCAD, Strasbourg, France, Tech. Rep (2010)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
* [J. Lamy, O. Merveille, B. Kerautret, N. Passat. "A benchmark framework for multi-region analysis of vesselness filters", 2022](https://hal.archives-ouvertes.fr/hal-03723493)
* [Sundararajan et al. , "Axiomatic Attribution for Deep Networks", 2017](https://arxiv.org/abs/1703.01365)

# TODO

- [ ] Implement of the MultiRes model
- [ ] Implement concurrency for attribution analysis
- [ ] Write unit tests
- [ ] Write anatomical_graph_to_image_graph in a more pythonic way (i.e. split the computation of coordinates from the function)
