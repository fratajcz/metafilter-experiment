import math
import time
import sys
import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict
import os

class MyJob(EvaluationJob):
    """ Ranking protocol for compound vs disease evaluation on Hetionet"""

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        #self.class_name = "CxDRankingHetionetAvgJob"
        self.model_name = config.get("model")
        self.dataset = config.get("user.dataset")
        print(self.dataset)
        if "hetionet" in self.dataset:
            fold = config.get("user.fold")
        self.device = self.config.get("job.device")
        dataset_variant = config.get("user.dataset_variant")
        self.base_path = "/home/icb/florin.ratajczak/kge/"

        #sys.path.append("/homestg/z0042eaf/")
        sys.path.append(self.base_path)
        from evaluator import Evaluator

        if self.eval_split == "valid":
            if "hetionet" in self.dataset:
                self.evaluator = Evaluator(ground_truth_train=os.path.join(self.base_path, "truth_hetionet/ground_truth_train_fold{}.npz".format(fold)),
                                           ground_truth_val=os.path.join(self.base_path, "truth_hetionet/ground_truth_val_fold{}.npz".format(fold)))
            elif "drkg" in self.dataset:
                self.evaluator = Evaluator(ground_truth_train=os.path.join(self.base_path,"truth_drkg/ground_truth_train.npz"),
                                           ground_truth_val=os.path.join(self.base_path,"truth_drkg/ground_truth_val.npz"))
        elif self.eval_split == "test":
            if "hetionet" in self.dataset:
                self.evaluator = Evaluator(ground_truth_train=os.path.join(self.base_path, "truth_hetionet/ground_truth_train_fold{}.npz".format(fold)),
                                           ground_truth_val=os.path.join(self.base_path,"truth_hetionet/ground_truth_test_fold{}.npz".format(fold)))
            elif "drkg" in self.dataset:
                self.evaluator = Evaluator(ground_truth_train=os.path.join(self.base_path,"truth_drkg/ground_truth_train.npz"),
                                           ground_truth_val=os.path.join(self.base_path,"truth_drkg/ground_truth_test.npz"))

        if "hetionet" in self.dataset:
            self.evaluation_relation_whitelist = ['CtD','CtD_inv']
        elif "drkg" in self.dataset:
            self.evaluation_relation_whitelist = ['union::treats::Compound:Disease', 'union::treats::Compound:Disease_inv']

        if "hetionet" in self.dataset:
            entity_ids_df = pd.read_csv(os.path.join(self.base_path, "data/hetionet-fold{}-{}/entity_ids.del".format(fold,dataset_variant)),sep="\t",names=['id', 'name'])
            relation_ids_df = pd.read_csv(os.path.join(self.base_path,"data/hetionet-fold{}-{}/relation_ids.del".format(fold,dataset_variant)),sep="\t",names=['id', 'name'])
        elif "drkg" in self.dataset:
            entity_ids_df = pd.read_csv(os.path.join(self.base_path,"data/drkg-{}/entity_ids.del".format(dataset_variant)),sep="\t",names=['id', 'name'])
            relation_ids_df = pd.read_csv(os.path.join(self.base_path,"data/drkg-{}/relation_ids.del".format(dataset_variant)),sep="\t",names=['id', 'name'])

        entity_ids = {x["name"]: x["id"] for _, x in entity_ids_df.iterrows()}
        relation_ids = {x["name"]: x["id"] for _, x in relation_ids_df.iterrows()}

        disease_encoder = preprocessing.LabelEncoder()
        compound_encoder = preprocessing.LabelEncoder()

        if "hetionet" in self.dataset:
            disease_encoder.classes_ = np.load(os.path.join(self.base_path, 'truth_hetionet/disease_classes_fold{}.npy'.format(fold)),allow_pickle = True)
            compound_encoder.classes_ = np.load(os.path.join(self.base_path,'truth_hetionet/compound_classes_fold{}.npy'.format(fold)),allow_pickle = True)
        elif "drkg" in self.dataset:
            disease_encoder.classes_ = np.load(os.path.join(self.base_path,'truth_drkg/disease_classes.npy'),allow_pickle = True)
            compound_encoder.classes_ = np.load(os.path.join(self.base_path,'truth_drkg/compound_classes.npy'),allow_pickle = True)


        self.disease_indices = torch.LongTensor([entity_ids[x] if x in entity_ids.keys() else -1 for x in disease_encoder.classes_]).to(self.device)
        self.compound_indices = torch.LongTensor([entity_ids[x] if x in entity_ids.keys() else -1 for x in compound_encoder.classes_]).to(self.device)
        self.relation_indices = torch.LongTensor([relation_ids[x] if x in relation_ids.keys() else -1 for x in self.evaluation_relation_whitelist]).to(self.device)


        self.num_diseases = self.disease_indices.shape[0]
        self.num_compounds = self.compound_indices.shape[0]
        self.num_relations = self.relation_indices.shape[0]


        missing_diseases = torch.sum(torch.where(self.disease_indices == -1, 1, 0))
        missing_compounds = torch.sum(torch.where(self.compound_indices == -1, 1, 0))
        missing_relations = torch.sum(torch.where(self.relation_indices == -1, 1, 0))

        self.config.print("Evaluation Dataset contains {} diseases, {} of which are not in the Training Set.".format(self.num_diseases,missing_diseases))
        self.config.print("Evaluation Dataset contains {} compounds, {} of which are not in the Training Set.".format(self.num_compounds,missing_compounds))
        self.config.print("Evaluation Dataset contains {} treats-edge(s), {} of which are not in the Training Set.".format(self.num_relations,missing_relations))

        random_entity = self.model.get_s_embedder().embed(self.disease_indices[0])
        self.entity_dim = random_entity.shape[0]

        random_relation = self.model.get_p_embedder().embed(self.relation_indices[0])
        self.relation_dim = random_relation.shape[0]
        """
        if self.model_name in ["reciprocal_relations_model","rescal"]:
            CxD_stack = []
            for s in self.compound_indices:
                for o in self.disease_indices:
                    CxD_stack.append([s, self.relation_indices[0], o])
            self.CxD_stack = torch.as_tensor(CxD_stack).to(self.device)
            test_score = self.model.score_spo(self.CxD_stack[0:2,0],self.CxD_stack[0:2,1],self.CxD_stack[0:2,2],"o")
            del CxD_stack

            DxC_stack = []
            for s in self.disease_indices:
                for o in self.compound_indices:
                    DxC_stack.append([s, self.relation_indices[1], o])
            self.DxC_stack = torch.as_tensor(DxC_stack).to(self.device)
            del DxC_stack
        """    
        
    @torch.no_grad()
    def _evaluate(self):

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        self.current_trace["epoch"] = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            epoch=self.epoch,
            batches=1,
            size=1,
            event="ranking_start"
        )

        if self.model_name not in ["reciprocal_relations_model","rescal"]:
            disease_embeddings = torch.stack([self.model.get_s_embedder().embed(x) if x != -1 else torch.zeros(self.entity_dim).to(self.device) for x in self.disease_indices])
            compound_embeddings = torch.stack([self.model.get_s_embedder().embed(x) if x != -1 else torch.zeros(self.entity_dim).to(self.device) for x in self.compound_indices])
            relation_embeddings = torch.stack([self.model.get_p_embedder().embed(x) if x != -1 else torch.zeros(self.relation_dim).to(self.device) for x in self.relation_indices])

        num_diseases = self.disease_indices.shape[0]
        num_compounds = self.compound_indices.shape[0]
        num_relations = self.relation_indices.shape[0]
        metrics = {}


        for i in range(num_relations):
            if self.evaluation_relation_whitelist[i] in ["CtD", 'union::treats::Compound:Disease']:
                if self.model_name in ["reciprocal_relations_model","rescal"]:                        
                    scores = torch.empty((self.num_compounds,self.num_diseases)).float().to(self.device)
                    for j in range(self.num_compounds):
                        compound = self.compound_indices[j]
                        compound_repeat = compound.repeat(self.num_diseases)
                        relation_repeat = self.relation_indices[i].repeat(self.num_diseases)
                        scores[j,:] = self.model.score_sp(compound_repeat,
                                                          relation_repeat,
                                                          self.disease_indices).float()

                    """
                    for j, s in enumerate(self.compound_indices):
                    
                        for k, o in enumerate(self.disease_indices):
                            scores[j,k] = self.model.score_sp(s.unsqueeze(0),
                                                              self.relation_indices[i].unsqueeze(0),
                                                              o.unsqueeze(0)).float()
                    """
                    """
                    for j in range(num_compounds):
                        scores[j,:] = self.model.score_sp(self.CxD_stack[num_diseases * j:num_diseases * (j +1),0],
                                                           self.CxD_stack[num_diseases * j:num_diseases * (j +1),1],
                                                           self.CxD_stack[num_diseases * j:num_diseases * (j +1),2])
                    """
                else: 
                    scores = self.model._scorer.score_emb(compound_embeddings, relation_embeddings[i].unsqueeze(0), disease_embeddings, combine= "sp_")
                scores = scores.reshape((self.num_compounds,self.num_diseases))
                                                    
            elif self.evaluation_relation_whitelist[i] in ["CtD_inv", 'union::treats::Compound:Disease_inv']:
                if self.model_name in ["reciprocal_relations_model","rescal"]:           
                    scores = torch.empty((num_diseases,num_compounds)).float().to(self.device)
                    for j in range(self.num_diseases):
                        disease = self.disease_indices[j]
                        disease_repeat = disease.repeat(self.num_compounds)
                        relation_repeat = self.relation_indices[i].repeat(self.num_compounds)
                        scores[j,:] = self.model.score_sp(disease_repeat,
                                                          relation_repeat,
                                                          self.compound_indices).float()
                    """
                    for j, s in enumerate(self.disease_indices):
                        for k, o in enumerate(self.compound_indices):
                            scores[j,k] = self.model.score_sp(s.unsqueeze(0),
                                                              self.relation_indices[i].unsqueeze(0),
                                                              o.unsqueeze(0)).float()
                    """                                          
                    """
                    for j in range(num_diseases):
                        scores[j,:] = self.model.score_sp(self.DxC_stack[num_compounds * j:num_compounds * (j +1),0],
                                                           self.DxC_stack[num_compounds * j:num_compounds * (j +1),1],
                                                           self.DxC_stack[num_compounds * j:num_compounds * (j +1),2])
                    """
                else:
                    scores = self.model._scorer.score_emb(disease_embeddings, relation_embeddings[i].unsqueeze(0), compound_embeddings, combine= "sp_")
                scores = scores.reshape((num_diseases,num_compounds)).t()


            self.evaluator.evaluate(scores.detach().cpu().numpy())
            if self.evaluation_relation_whitelist[i] in ["CtD", 'union::treats::Compound:Disease']:
                metrics.update({"mrr_CxD_train": float(self.evaluator.mrrs_row_train[-1]),
                                "mrr_CxD_val": float(self.evaluator.mrrs_row_val[-1])})
                    
            else:
                metrics.update({"mrr_DxC_train": float(self.evaluator.mrrs_col_train[-1]),
                                "mrr_DxC_val": float(self.evaluator.mrrs_col_val[-1])})
        

        metrics.update({"mrr_avg_val": (metrics["mrr_CxD_val"] + metrics["mrr_DxC_val"]) / 2})

        self.current_trace["epoch"].update(event="eval_completed", **metrics, device=self.device)
