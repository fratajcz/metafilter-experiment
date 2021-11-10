# Purpose

This repository contains the code for the experiments in the paper "Task-Driven Knowledge Graph Filtering Improves Prioritizing Drugs for Repurposing" (DOI: [10.21203/rs.3.rs-721705/v1](https://doi.org/10.21203/rs.3.rs-721705/v1). 

# Installation

We have made extensive use of the LibKGE framework. To make sure that we measure only the performance in our task during performance evaluation, we have added a Job that is specific to the experiments in the paper. It scores all possible relations between compounds and diseases and nothing else. 

## Install LibKGE

Go to https://github.com/uma-pi1/kge and follow their installation instructions. Required is version 0.1, which is the most recent version at the time of writing this.

## Place MyFile Directory

Place the files in this repo's ```job``` dir (MyFile.py, MyFile.yaml) in the libkge directory ```kge/kge/job/```. Add the line 
```
from kge.job.MyFile import MyJob
```
to the libkge file ```kge/kge/job/__init__.py```.

you can now import the job MyJob in your libkge config files using ```import: [MyFile]``` and use the following lines in your config file:

```
eval:
  type: MyJob
```

However, this won't work yet because it needs the evaluator class that handles the actual Compound vs Disease evaluations.

## Place Evaluator

Place the evaluator.py module in your ```kge/```repository so it can be used by the respective evaluation jobs plugins.

## Place ground truths

To reproduce the results in the paper with the exact same splits, you can use the ground truth files given in this repo. They contain the Compound x Disease adjacency matrices and make sure that all runs - especially original and modified version of datasets - use the same ground truth to evaluate on, since one of them obviously contains more edges than the other. Place these directories (the whole directories (```truth_hetionet, thruth_drkg```), not just the contents) in the libkge directory ```kge/```, on the same level where you also put the evaluator script. 

If you use your own dataset and/or produced new splits of Hetionet and/or DRKG, check the Usage section on how to produce you own ground truth files.

## Place datasets

Download the datasets from [10.5281/zenodo.5638999](https://doi.org/10.5281/zenodo.5638999), unzip them (i.e. using ```tar -xzvf data data.tar.gz```) and place the contents in ```kge/data/```. Note that there should not be any intermediate directories like ```kge/data/whatever/another/hetionet-fold1-subset-with-inverse/``` but instead just ```kge/data/hetionet-fold1-subset-with-inverse/``` to work.

# Usage

## training

To recreate the experiments detailed in the paper, just use on of our config files in the ```recipes/``` directory. The filenames should make it pretty clear which experiment they contain. Since the HPO might be very time-consuming on single-GPU runs, you are free to use multiple GPUs using LibKGEs excellent parallelization scheme:

```
kge start recipes/hpo-CxD-hetionet-fold1-subset-with-inverse-complex-both.yaml --search.device_pool cuda:0,cuda:1 --search.num_workers 2
```

This line uses two GPUs, but you could easily extend this to as many GPUs as you like.

## testing

For testing we want to gather more metrics than the MRR, which is used for the hyperparameter search, and use a different holdout set.
Therefore, find the path to the folder that contains your best model's best checkpoint. If you have conducted the HPO as detailed above, you can check the ```kge.log```file in the HPO's main directory, which should be located somewhere in ```kge/local/experiments/```and start with a timestamp. At the end of this logfile you should find a line that reads something like ```Best trial (00027): mrr_avg_val=0.21183539548018415```, depending on which was the best trial and what was the best performance, of course. So now you know that the path you need for testing is ```kge/local/experiments/[your-experiment-name]/00027/```, which should contain a file named```checkpoint_best.pt```. 

So now that you know the path to your best runs best checkpoint, just run:

```
python3 evaluate_test.py -p kge/local/experiments/[your-experiment-name]/00027/
```

without adding ```checkpoint_best.pt```. This create a ```results``` directory in you ```kge```directory containing the test results for the experiment that you provided. Note that all the paths that you have to use here depend on your directory structure and might be different on your machine!

# Use it on your own dataset!

To see if your dataset can benefit from task-driven modification, check the sister repo [metafilter-apply](https://github.com/fratajcz/metafilter-apply) and read the instructions on how to apply metapath based filtering to your Knowledge Graph!