from kge.model import KgeModel
from kge.util.io import load_checkpoint
import torch
import sys
import pandas as pd 
import sklearn.preprocessing as preprocessing
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str, default="", help="path to the directory where checkpoint_best.pt is stored.")
args = parser.parse_args()

#best_model = "local/experiments/20210201-181752-hpo-CxD-hetionet-fold1-subset-with-inverse-transe-both/00014/checkpoint_best.pt"

prefix = "/aig/users/z0042eaf/experiments/"
#prefix = "/homestg/z0042eaf/src/libkge/local/experiments/"
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
best_model = prefix + run + suffix

if "hetionet" in run:
    evaluation_relation_whitelist = ['CtD',"CtD_inv"]
elif "drkg" in run:
    evaluation_relation_whitelist = ['union::treats::Compound:Disease', 'union::treats::Compound:Disease_inv']
else:
    raise ValueError(best_model)



#best_model = "/homestg/z0042eaf/src/libkge/local/experiments/20210205-071150-hpo-default-hetionet-fold1-subset-with-inverse-conve-both/00000/checkpoint_best.pt"

checkpoint = load_checkpoint(best_model)

if "full" in best_model:
    dataset_version = "full"
elif "subset" in best_model:
    dataset_version = "subset"
elif "yushan" in best_model:
    dataset_version = ""
else:
    raise ValueError(best_model)

#if "inverse" in best_model:
dataset_version += "-with-inverse"

if "hetionet" in best_model:
    if "fold1" in best_model:
        fold = 1
    elif "fold2" in best_model:
        fold = 2
    elif "fold3" in best_model:
        fold = 3
    elif "fold4" in best_model:
        fold = 4
    elif "yushan" in best_model:
        fold = "yushan"
    else:
        raise ValueError(best_model)

model = KgeModel.create_from(checkpoint)

model = model

sys.path.append("/homestg/z0042eaf/")
from evaluator import Evaluator

if "hetionet" in best_model:
    evaluator = Evaluator(ground_truth_train="/homestg/z0042eaf/ground_truth_train_fold{}.npz".format(fold),
                          ground_truth_val="/homestg/z0042eaf/ground_truth_val_fold{}.npz".format(fold),
                          ground_truth_test="/homestg/z0042eaf/ground_truth_test_fold{}.npz".format(fold))
elif "drkg" in best_model:
    evaluator = Evaluator(ground_truth_train="/homestg/z0042eaf/truth_drkg/ground_truth_train.npz",
                          ground_truth_val="/homestg/z0042eaf/truth_drkg/ground_truth_val.npz",
                          ground_truth_test="/homestg/z0042eaf/truth_drkg/ground_truth_test.npz")

if "hetionet" in best_model:
    if dataset_version != "":
        entity_ids_df = pd.read_csv("/homestg/z0042eaf/src/libkge/data/hetionet-fold{}-{}/entity_ids.del".format(fold,dataset_version),sep="\t",names=['id', 'name'])
        relation_ids_df = pd.read_csv("/homestg/z0042eaf/src/libkge/data/hetionet-fold{}-{}/relation_ids.del".format(fold,dataset_version),sep="\t",names=['id', 'name'])
    else:
        entity_ids_df = pd.read_csv("/home/fratajczak/kge/data/hetionet-{}/entity_ids.del".format(fold),sep="\t",names=['id', 'name'])
        relation_ids_df = pd.read_csv("/home/fratajczak/kge/data/hetionet-{}/relation_ids.del".format(fold),sep="\t",names=['id', 'name'])
elif "drkg" in best_model:
    entity_ids_df = pd.read_csv("/homestg/z0042eaf/src/libkge/data/drkg-{}/entity_ids.del".format(dataset_version),sep="\t",names=['id', 'name'])
    relation_ids_df = pd.read_csv("/homestg/z0042eaf/src/libkge/data/drkg-{}/relation_ids.del".format(dataset_version),sep="\t",names=['id', 'name'])

entity_ids = {x["name"]: x["id"] for _, x in entity_ids_df.iterrows()}
relation_ids = {x["name"]: x["id"] for _, x in relation_ids_df.iterrows()}

disease_encoder = preprocessing.LabelEncoder()
compound_encoder = preprocessing.LabelEncoder()

if "hetionet" in best_model:
    disease_encoder.classes_ = np.load('/homestg/z0042eaf/disease_classes_fold{}.npy'.format(fold),allow_pickle = True)
    compound_encoder.classes_ = np.load('/homestg/z0042eaf/compound_classes_fold{}.npy'.format(fold),allow_pickle = True)
elif "drkg" in best_model:
    disease_encoder.classes_ = np.load('/homestg/z0042eaf/truth_drkg/disease_classes.npy',allow_pickle = True)
    compound_encoder.classes_ = np.load('/homestg/z0042eaf/truth_drkg/compound_classes.npy',allow_pickle = True)

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



disease_embeddings = torch.stack([model.get_s_embedder().embed(x) if x != -1 else torch.zeros(entity_dim) for x in disease_indices])
compound_embeddings = torch.stack([model.get_s_embedder().embed(x) if x != -1 else torch.zeros(entity_dim) for x in compound_indices])
relation_embeddings = torch.stack([model.get_p_embedder().embed(x) if x != -1 else torch.zeros(relation_dim) for x in relation_indices])

num_diseases = disease_embeddings.shape[0]
num_compounds = compound_embeddings.shape[0]
num_relations = relation_embeddings.shape[0]

metrics = {}

for i in range(num_relations):
    if evaluation_relation_whitelist[i] in ["CtD", 'union::treats::Compound:Disease']:
        if "conve" in best_model or "rescal" in best_model:
            stack = []
            for s in compound_indices:
                for o in disease_indices:
                    stack.append([s, relation_indices[i], o])
            stack = torch.as_tensor(stack).cuda()
            
            scores = torch.empty((num_compounds,num_diseases)).cuda()
            for j in range(num_compounds):
                    scores[j,:] = model.score_spo(stack[num_diseases * j:num_diseases * (j +1),0],
                                                  stack[num_diseases * j:num_diseases * (j +1),1],
                                                  stack[num_diseases * j:num_diseases * (j +1),2], "o")

        else:
            scores = model._scorer.score_emb(compound_embeddings, relation_embeddings[i].unsqueeze(0), disease_embeddings, combine= "sp_")
        scores = scores.reshape((num_compounds,num_diseases))

    else:
        if "conve" in best_model or "rescal" in best_model:
            stack = []
            for s in disease_indices:
                for o in compound_indices:
                    stack.append([s, relation_indices[i], o])
            stack = torch.as_tensor(stack)
            scores = torch.empty((num_diseases,num_compounds))
            for j in range(num_diseases):
                    scores[j,:] = model.score_spo(stack[num_compounds * j:num_compounds * (j +1),0],
                                                  stack[num_compounds * j:num_compounds * (j +1),1],
                                                  stack[num_compounds * j:num_compounds * (j +1),2], "o")

        else:
            scores = model._scorer.score_emb(disease_embeddings, relation_embeddings[i].unsqueeze(0), compound_embeddings, combine= "sp_")
            
        scores = scores.reshape((num_diseases,num_compounds)).t()


    evaluator.evaluate(scores.detach().cpu().numpy(), use_testing=True)
    if evaluation_relation_whitelist[i] in ["CtD", 'union::treats::Compound:Disease']:
        metrics.update({"mrr_CxD_train": float(evaluator.mrrs_row_train[-1]),
                        "mrr_CxD_val": float(evaluator.mrrs_row_val[-1]),
                        "mrr_CxD_test": float(evaluator.mrrs_row_test[-1])})  
    else:
        metrics.update({"mrr_DxC_train": float(evaluator.mrrs_col_train[-1]),
                        "mrr_DxC_val": float(evaluator.mrrs_col_val[-1]),
                        "mrr_DxC_test": float(evaluator.mrrs_col_test[-1])})


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
evaluator.random(10,seed=1,proximity=scores.detach().cpu().numpy(), use_testing=True)
