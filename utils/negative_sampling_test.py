from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.util import KgeSampler
from kge import Dataset, Config
import torch
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import entropy


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", type=str, default="", help="path to the directory where checkpoint_best.pt is stored.")
args = parser.parse_args()

prefix = "/aig/users/z0042eaf/experiments/"
#prefix = "/homestg/z0042eaf/src/libkge/local/experiments/"
suffix = "checkpoint_best.pt"

num_draws = 100

#DRKG:
##Complex:
#run = {"subset": "20210302-170815-hpo-CxD-drkg-subset-with-inverse-complex-both/00006/",
#        "full": "20210302-171854-hpo-CxD-drkg-full-with-inverse-complex-both/00006/"}

# Hetionet:
## RESCAL:

#run = {"subset": "20210212-062234-hpo-CxD-hetionet-fold1-subset-with-inverse-rescal-both/00019/",
#        "full": "20210212-062234-hpo-CxD-hetionet-fold1-full-with-inverse-rescal-both/00019/"}

#ComplEx
#run = {"subset": "20210203-064415-hpo-CxD-hetionet-fold1-subset-with-inverse-complex-both/00006/",
#        "full": "20210203-064415-hpo-CxD-hetionet-fold1-full-with-inverse-complex-both/00006/"}

runs = {"hetionet": 
        {"full": "20210203-064415-hpo-CxD-hetionet-fold1-full-with-inverse-complex-both/00006/",
        "subset": "20210203-064415-hpo-CxD-hetionet-fold1-subset-with-inverse-complex-both/00006/"},
        "drkg":
         {"full": "20210302-171854-hpo-CxD-drkg-full-with-inverse-complex-both/00006/",
         "subset": "20210302-170815-hpo-CxD-drkg-subset-with-inverse-complex-both/00006/"}
        }

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


cm = 1/2.54  # centimeters in inches

fig, axes = plt.subplots(2,2,figsize=(17*cm, 15*cm), dpi=400)

for i, (dataset_name, run) in enumerate(runs.items()):
    for j, version in enumerate(run.keys()):
        print(version)

        best_model = prefix + run[version] + suffix

        checkpoint = load_checkpoint(best_model)

        model = KgeModel.create_from(checkpoint)
        dataset_config = Config()
        
        if dataset_name == "hetionet":
            dataset_config.load("/home/z0042eaf/src/libkge/hpo-CxD-hetionet-fold1-{}-with-inverse-complex-both.yaml".format(version))
        else:
            dataset_config.load("/home/z0042eaf/src/libkge/hpo-CxD-drkg-{}-with-inverse-complex-both.yaml".format(version))
        dataset = Dataset.create(dataset_config)
        print("Dataset Loaded")

        ns_config = Config()
        ns_config.load(filename="/homestg/z0042eaf/src/libkge/negative_sampling_test.yaml")
        sampler = KgeSampler.create(ns_config, "negative_sampling", dataset)
        print("Sampler initialized")
        entropies = []
        pos_scores_collection = []
        neg_scores_collection = []

        for draw in range(num_draws):
            if draw % 10 == 0:
                print(".", end="", flush=True)

            batch = torch.randint(low=0, high=dataset.num_relations(), size=(2000,))
            triples = dataset.split("train")[batch, :].long()

            negative_samples = list()
            negative_samples.append(sampler.sample(triples, 2))

            pos_scores = model.score_spo(
                        triples[:, 0], triples[:, 1], triples[:, 2], direction="o",
                    ).detach().numpy()

            neg_scores = negative_samples[0].score(
                        model
                    ).flatten().detach().numpy()


            min = np.min((pos_scores,neg_scores))

            pos_scores = pos_scores + np.abs(min) + 1e-16
            neg_scores = neg_scores + np.abs(min) + 1e-16


            max = np.max((pos_scores,neg_scores))


            pos_scores = pos_scores / max 
            neg_scores = neg_scores / max

            entropies.append(entropy(pos_scores,neg_scores))
            pos_scores_collection.append(pos_scores)
            neg_scores_collection.append(neg_scores)

        print("")
        print("{}: {}".format(version, np.mean(entropies)))

        pos_scores_all = np.append(pos_scores_collection[0],pos_scores_collection[1:])
        neg_scores_all = np.append(neg_scores_collection[0],neg_scores_collection[1:])
        bins = np.linspace(0, 1, 50)

        

        axes[i][j].hist(pos_scores_all, bins=bins, alpha=0.5, label='positive')
        axes[i][j].hist(neg_scores_all, bins=bins, alpha=0.5, label='negative')
        axes[i][j].set_xlabel("Probability")
        axes[i][j].set_ylabel("Count")
        del pos_scores_all, neg_scores_all, neg_scores_collection, pos_scores_collection

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("/homestg/z0042eaf/negative_sampling_test_{}_both.png".format(num_draws))
plt.clf()
#print(entropy(pos_scores,neg_scores))