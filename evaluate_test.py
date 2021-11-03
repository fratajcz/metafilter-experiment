from kge.model import KgeModel
from kge.util.io import load_checkpoint
import torch
import sys
import pandas as pd 
import sklearn.preprocessing as preprocessing
import numpy as np
import os
from evaluator import Evaluator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str, default="", help="path to the directory where checkpoint_best.pt is stored.")
args = parser.parse_args()


prefix = ""
suffix = "checkpoint_best.pt"

#### Best Checkpoints for the best Runs:
#Fold1
#DistMult
#Subset: 
#run = "20210203-062208-hpo-CxD-hetionet-fold1-subset-with-inverse-distmult-both/00006/"
#Full:
#run = "20210203-062637-hpo-CxD-hetionet-fold1-full-with-inverse-distmult-both/00006/"
##Default Eval:
#Subset:
#run = "20210203-082737-hpo-default-hetionet-fold1-subset-with-inverse-distmult-both/00006/"
#Full:
#run = "20210203-082636-hpo-default-hetionet-fold1-full-with-inverse-distmult-both/00006/"

#ComplEx
# Subset:
#run = "20210203-064415-hpo-CxD-hetionet-fold1-subset-with-inverse-complex-both/00006/"
# Full:
#run = "20210203-064415-hpo-CxD-hetionet-fold1-full-with-inverse-complex-both/00006/"
##Default Eval:
# Subset:
#run = "20210203-083107-hpo-default-hetionet-fold1-subset-with-inverse-complex-both/00006/"
# Full:
#run = "20210203-083135-hpo-default-hetionet-fold1-full-with-inverse-complex-both/00006/"

#Rescal
# Subset:
#run = "20210212-062234-hpo-CxD-hetionet-fold1-subset-with-inverse-rescal-both/00019/"
# Full:
#run = "20210212-062234-hpo-CxD-hetionet-fold1-full-with-inverse-rescal-both/00019/"
# Default Eval:
#Subset:
#run = "20210212-090048-hpo-default-hetionet-fold1-subset-with-inverse-rescal-both/00019/"
# Full:
#run =  "20210212-092326-hpo-default-hetionet-fold1-full-with-inverse-rescal-both/00008/"

#ConvE:
# Subset:
#run = "20210212-092340-hpo-CxD-hetionet-fold1-subset-with-inverse-conve-both/00025/"
# Full:
#run = "20210212-071051-hpo-CxD-hetionet-fold1-full-with-inverse-conve-both/00018/"
# Default Eval:
# Subset:
#run = "20210212-084216-hpo-default-hetionet-fold1-subset-with-inverse-conve-both/00004/"
#Full:
#run =  "20210212-080841-hpo-default-hetionet-fold1-full-with-inverse-conve-both/00008/"


#DRKG:
##Complex:
# Subset:
#run = "20210302-170815-hpo-CxD-drkg-subset-with-inverse-complex-both/00006/"
# Full:
run = "20210302-171854-hpo-CxD-drkg-full-with-inverse-complex-both/00006/"

## DistMult:
# Subset:
#run = "20210302-113022-hpo-CxD-drkg-subset-with-inverse-distmult-both/00006/"
# Full:
#run = "20210302-154646-hpo-CxD-drkg-full-with-inverse-distmult-both/00006/"

## TransE:
# Subset:
#run =  "20210302-082755-hpo-CxD-drkg-subset-with-inverse-transe-both/00026/"
# Full:
#run = "20210308-092854-hpo-CxD-drkg-full-with-inverse-transe-both/00026/"

## RESCAL:
# Subset:
#run = "20210308-115223-train-19-CxD-drkg-subset-rescal-both/"
# Full:
# run = "20210308-115214-train-19-CxD-drkg-full-rescal-both/"

## ConvE:
# Subset: 
# run = 20210301-095554-train-24-CxD-drkg-subset-conve-both/  
# Full:
# run = 20210301-094420-train-18-CxD-drkg-full-conve-both/

if args.path != "":
    run = args.path

print(run)
best_model = os.path.join(prefix, run, suffix)

if "hetionet" in run:
    evaluation_relation_whitelist = ['CtD',"CtD_inv"]
elif "drkg" in run:
    evaluation_relation_whitelist = ['union::treats::Compound:Disease', 'union::treats::Compound:Disease_inv']
else:
    raise ValueError(best_model)


checkpoint = load_checkpoint(best_model)



if "full" in best_model:
    dataset_version = "full"
elif "subset" in best_model:
    dataset_version = "subset"
elif "noana" in best_model:
    dataset_version = "noana"
elif "nogene" in best_model:
    dataset_version = "nogene"
elif "nosymp" in best_model:
    dataset_version = "nosymp"
elif "noside" in best_model:
    dataset_version = "noside"
elif "nopc" in best_model:
    dataset_version = "nopc"
else:
    raise ValueError(best_model)

#if "inverse" in best_model:
dataset_version += "-with-inverse"

if "hetionet" in best_model:
    dataset = "hetionet"
elif "drkg" in best_model:
    dataset = "drkg"


if "hetionet" in best_model:
    if "fold1" in best_model:
        fold = 1
    elif "fold2" in best_model:
        fold = 2
    elif "fold3" in best_model:
        fold = 3
    elif "fold4" in best_model:
        fold = 4
    else:
        raise ValueError(best_model)


checkpoint["config"].set("dataset.name","{}-fold{}-{}".format(dataset,fold,dataset_version))

model = KgeModel.create_from(checkpoint)

model = model



base_path= "./kge"

if "hetionet" in best_model:
    evaluator = Evaluator(ground_truth_train=os.path.join(base_path, "truth_hetionet/ground_truth_train_fold{}.npz".format(fold)),
                          ground_truth_val=os.path.join(base_path, "truth_hetionet/ground_truth_val_fold{}.npz".format(fold)),
                          ground_truth_test=os.path.join(base_path, "truth_hetionet/ground_truth_test_fold{}.npz".format(fold)))
elif "drkg" in best_model:
    evaluator = Evaluator(ground_truth_train=os.path.join(base_path,"truth_drkg/ground_truth_train.npz"),
                          ground_truth_val=os.path.join(base_path,"truth_drkg/ground_truth_val.npz"),
                          ground_truth_test=os.path.join(base_path,"truth_drkg/ground_truth_test.npz"))

if "hetionet" in best_model:
    if dataset_version != "":
        entity_ids_df = pd.read_csv(os.path.join(base_path,"data/hetionet-fold{}-{}/entity_ids.del".format(fold, dataset_version)),sep="\t",names=['id', 'name'])
        relation_ids_df = pd.read_csv(os.path.join(base_path,"data/hetionet-fold{}-{}/relation_ids.del".format(fold, dataset_version)),sep="\t",names=['id', 'name'])
    else:
        entity_ids_df = pd.read_csv("/home/fratajczak/kge/data/hetionet-{}/entity_ids.del".format(fold),sep="\t",names=['id', 'name'])
        relation_ids_df = pd.read_csv("/home/fratajczak/kge/data/hetionet-{}/relation_ids.del".format(fold),sep="\t",names=['id', 'name'])
elif "drkg" in best_model:
    entity_ids_df = pd.read_csv(os.path.join(base_path,"data/drkg-{}/entity_ids.del".format(dataset_version)),sep="\t",names=['id', 'name'])
    relation_ids_df = pd.read_csv(os.path.join(base_path,"data/drkg-{}/relation_ids.del".format(dataset_version)),sep="\t",names=['id', 'name'])

entity_ids = {x["name"]: x["id"] for _, x in entity_ids_df.iterrows()}
relation_ids = {x["name"]: x["id"] for _, x in relation_ids_df.iterrows()}

disease_encoder = preprocessing.LabelEncoder()
compound_encoder = preprocessing.LabelEncoder()

if "hetionet" in best_model:
    disease_encoder.classes_ = np.load(os.path.join(base_path, 'truth_hetionet/disease_classes_fold{}.npy'.format(fold)),allow_pickle = True)
    compound_encoder.classes_ = np.load(os.path.join(base_path, 'truth_hetionet/compound_classes_fold{}.npy'.format(fold)),allow_pickle = True)
elif "drkg" in best_model:
    disease_encoder.classes_ = np.load(os.path.join(base_path, 'truth_drkg/disease_classes.npy'),allow_pickle = True)
    compound_encoder.classes_ = np.load(os.path.join(base_path, 'truth_drkg/compound_classes.npy'),allow_pickle = True)

disease_indices = torch.LongTensor([entity_ids[x] if x in entity_ids.keys() else -1 for x in disease_encoder.classes_])
compound_indices = torch.LongTensor([entity_ids[x] if x in entity_ids.keys() else -1 for x in compound_encoder.classes_])
relation_indices = torch.LongTensor([relation_ids[x] if x in relation_ids.keys() else -1 for x in evaluation_relation_whitelist])
 
missing_diseases = torch.sum(torch.where(disease_indices == -1, 1, 0))
missing_compounds = torch.sum(torch.where(compound_indices == -1, 1, 0))
missing_relations = torch.sum(torch.where(relation_indices == -1, 1, 0))

print("Evaluation Dataset contains {} diseases, {} of which are not in the Training Set.".format(disease_indices.shape[0],missing_diseases))
print("Evaluation Dataset contains {} compounds, {} of which are not in the Training Set.".format(compound_indices.shape[0],missing_compounds))
print("Evaluation Dataset contains {} treats-edge(s), {} of which are not in the Training Set.".format(relation_indices.shape[0],missing_relations))

random_entity = model.get_s_embedder().embed(disease_indices[0])
entity_dim = random_entity.shape[0]

random_relation = model.get_p_embedder().embed(relation_indices[0])
relation_dim = random_relation.shape[0]


if "conve" not in best_model and "rescal" not in best_model:
    disease_embeddings = torch.stack([model.get_s_embedder().embed(x) if x != -1 else torch.zeros(entity_dim) for x in disease_indices])
    compound_embeddings = torch.stack([model.get_s_embedder().embed(x) if x != -1 else torch.zeros(entity_dim) for x in compound_indices])
    relation_embeddings = torch.stack([model.get_p_embedder().embed(x) if x != -1 else torch.zeros(relation_dim) for x in relation_indices])

num_diseases = disease_indices.shape[0]
num_compounds = compound_indices.shape[0]
num_relations = relation_indices.shape[0]

metrics = {}

for i in range(num_relations):
    if evaluation_relation_whitelist[i] in ["CtD", 'union::treats::Compound:Disease']:
        if "conve" in best_model or "rescal" in best_model:
            
            scores = torch.empty((num_compounds,num_diseases))

            relation = relation_indices[i]
            for j, s in enumerate(compound_indices):
                for k, o in enumerate(disease_indices):
                    scores[j,k] = model.score_sp(s.unsqueeze(0),
                                              relation.unsqueeze(0),
                                              o.unsqueeze(0))

        else:
            scores = model._scorer.score_emb(compound_embeddings, relation_embeddings[i].unsqueeze(0), disease_embeddings, combine= "sp_")
        scores = scores.reshape((num_compounds,num_diseases))

    else:
        if "conve" in best_model or "rescal" in best_model:
            scores = torch.empty((num_diseases,num_compounds))
            relation = relation_indices[i]
            for j, s in enumerate(disease_indices):
                for k, o in enumerate(compound_indices):
                    scores[j,k] = model.score_sp(s.unsqueeze(0),
                                              relation.unsqueeze(0),
                                              o.unsqueeze(0))

        else:
            scores = model._scorer.score_emb(disease_embeddings, relation_embeddings[i].unsqueeze(0), compound_embeddings, combine= "sp_")
            
        scores = scores.reshape((num_diseases,num_compounds)).t()


    evaluator.evaluate(scores.detach().cpu().numpy(), use_testing=True)

    if evaluation_relation_whitelist[i] in ["CtD", 'union::treats::Compound:Disease']:
        metrics.update({"mrr_CxD_train": float(evaluator.mrrs_row_train[-1]),
                        "mrr_CxD_val": float(evaluator.mrrs_row_val[-1]),
                        "mrr_CxD_test": float(evaluator.mrrs_row_test[-1])})  
        metrics.update({"mean_rank_CxD_train": float(evaluator.mean_ranks_row_train[-1]),
                        "mean_rank_CxD_val": float(evaluator.mean_ranks_row_val[-1]),
                        "mean_rank_CxD_test": float(evaluator.mean_ranks_row_test[-1])}) 
        metrics.update({"hat5_CxD_train": float(evaluator.hat5_row_train[-1]),
                        "hat5_CxD_val": float(evaluator.hat5_row_val[-1]),
                        "hat5_CxD_test": float(evaluator.hat5_row_test[-1])}) 
        metrics.update({"hat10_CxD_train": float(evaluator.hat10_row_train[-1]),
                        "hat10_CxD_val": float(evaluator.hat10_row_val[-1]),
                        "hat10_CxD_test": float(evaluator.hat10_row_test[-1])}) 
        metrics.update({"hat20_CxD_train": float(evaluator.hat20_row_train[-1]),
                        "hat20_CxD_val": float(evaluator.hat20_row_val[-1]),
                        "hat20_CxD_test": float(evaluator.hat20_row_test[-1])}) 
        metrics.update({"hat50_CxD_train": float(evaluator.hat50_row_train[-1]),
                        "hat50_CxD_val": float(evaluator.hat50_row_val[-1]),
                        "hat50_CxD_test": float(evaluator.hat50_row_test[-1])}) 
    else:
        metrics.update({"mrr_DxC_train": float(evaluator.mrrs_col_train[-1]),
                        "mrr_DxC_val": float(evaluator.mrrs_col_val[-1]),
                        "mrr_DxC_test": float(evaluator.mrrs_col_test[-1])})
        metrics.update({"mean_rank_DxC_train": float(evaluator.mean_ranks_col_train[-1]),
                        "mean_rank_DxC_val": float(evaluator.mean_ranks_col_val[-1]),
                        "mean_rank_DxC_test": float(evaluator.mean_ranks_col_test[-1])}) 
        metrics.update({"hat5_DxC_train": float(evaluator.hat5_col_train[-1]),
                        "hat5_DxC_val": float(evaluator.hat5_col_val[-1]),
                        "hat5_DxC_test": float(evaluator.hat5_col_test[-1])}) 
        metrics.update({"hat10_DxC_train": float(evaluator.hat10_col_train[-1]),
                        "hat10_DxC_val": float(evaluator.hat10_col_val[-1]),
                        "hat10_DxC_test": float(evaluator.hat10_col_test[-1])}) 
        metrics.update({"hat20_DxC_train": float(evaluator.hat20_col_train[-1]),
                        "hat20_DxC_val": float(evaluator.hat20_col_val[-1]),
                        "hat20_DxC_test": float(evaluator.hat20_col_test[-1])}) 
        metrics.update({"hat50_DxC_train": float(evaluator.hat50_col_train[-1]),
                        "hat50_DxC_val": float(evaluator.hat50_col_val[-1]),
                        "hat50_DxC_test": float(evaluator.hat50_col_test[-1])}) 

metric_keys = ["mrr","mean_rank","hat5","hat10","hat20","hat50"]
splits = ["train","val","test"]

for metric in metric_keys:
    for split in splits:
        CxD = metrics["{}_CxD_{}".format(metric, split)]
        DxC = metrics["{}_DxC_{}".format(metric, split)]

        mean = np.mean((CxD, DxC))
        metrics.update({"{}_both_{}".format(metric, split): mean})

models = ["rescal","complex","transe","conve","distmult"]

for model in models:
    if model in best_model:
        actual_model = model

results = pd.DataFrame(data = metrics.values(), index=metrics.keys(), columns=["{}-{}-{}.csv".format(dataset,dataset_version,actual_model)])

#results.to_csv(os.path.join(base_path, "results","hetionet-crossval", "{}-{}-{}.csv".format(dataset,dataset_version,actual_model)))
if "hetionet" in best_model:
    results.to_csv(os.path.join(base_path, "results", "{}-fold{}-{}-{}.csv".format(dataset,fold,dataset_version,actual_model)))
else:
    results.to_csv(os.path.join(base_path, "results", "{}-{}-{}.csv".format(dataset,dataset_version,actual_model)))

print(metrics)

##### Manually compute compound - CtD - disease ~ 0:
"""
compound_minus_relation =  torch.subtract(compound_embeddings, relation_embeddings[0])

cdist_result = torch.cdist(compound_minus_relation,disease_embeddings,1)
proximity_matrix = cdist_result * -1


evaluator.evaluate(proximity_matrix.detach().numpy(), use_testing=True)


metrics = {"mrr_CxD_train": float(evaluator.mrrs_row_train[-1]),
            "mrr_CxD_val": float(evaluator.mrrs_row_val[-1]),
            "mrr_CxD_test": float(evaluator.mrrs_row_test[-1]),
            "mrr_DxC_train": float(evaluator.mrrs_col_train[-1]),
            "mrr_DxC_val": float(evaluator.mrrs_col_val[-1]),
            "mrr_DxC_test": float(evaluator.mrrs_col_test[-1])}

print("Compound - {} = Disease:".format(evaluation_relation_whitelist[0]))
print(metrics)

##### Manually compute compound + CtD - disease ~ 0:

compound_plus_relation =  torch.add(compound_embeddings, relation_embeddings[0])

cdist_result = torch.cdist(compound_plus_relation,disease_embeddings,1)
proximity_matrix = cdist_result * -1

evaluator.evaluate(proximity_matrix.detach().numpy(), use_testing=True)


metrics = {"mrr_CxD_train": float(evaluator.mrrs_row_train[-1]),
            "mrr_CxD_val": float(evaluator.mrrs_row_val[-1]),
            "mrr_CxD_test": float(evaluator.mrrs_row_test[-1]),
            "mrr_DxC_train": float(evaluator.mrrs_col_train[-1]),
            "mrr_DxC_val": float(evaluator.mrrs_col_val[-1]),
            "mrr_DxC_test": float(evaluator.mrrs_col_test[-1])}


print("Compound + {} = Disease:".format(evaluation_relation_whitelist[0]))
print(metrics)
"""
#evaluator.random(10,seed=1,proximity=scores.detach().cpu().numpy(), use_testing=True)
