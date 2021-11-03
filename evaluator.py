import numpy as np
from scipy.sparse import load_npz

class Evaluator():
    ''' Class to perform the Compound-Disease performance evaluation. 
        By convention the compounds are the rows and the diseases the columns of the matrices
        '''
    def __init__(self,ground_truth_train=None,ground_truth_val=None,ground_truth_test=None):
        '''Truth matrices can be passed when the class is initialized or set explicitely later'''
        self.mrrs_row_train = []
        self.mrrs_row_val = []
        self.mrrs_col_train = []
        self.mrrs_col_val = []

        self.mean_ranks_row_train = []
        self.mean_ranks_row_val = []
        self.mean_ranks_col_train = []
        self.mean_ranks_col_val = []


        self.hat5_row_train = []
        self.hat5_row_val = []
        self.hat5_col_train = []
        self.hat5_col_val = []


        self.hat10_row_train = []
        self.hat10_row_val = []
        self.hat10_col_train = []
        self.hat10_col_val = []


        self.hat20_row_train = []
        self.hat20_row_val = []
        self.hat20_col_train = []
        self.hat20_col_val = []

        self.hat50_row_train = []
        self.hat50_row_val = []
        self.hat50_col_train = []
        self.hat50_col_val = []

        self.evaluation_epochs = []
        self.truth_val_matrix = load_npz(ground_truth_val).toarray() if ground_truth_val != None else None
        self.truth_train_matrix = load_npz(ground_truth_train).toarray() if ground_truth_train != None else None
        self.truth_test_matrix = load_npz(ground_truth_test).toarray() if ground_truth_test != None else None

        if self.truth_test_matrix is not None and self.truth_train_matrix is not None:
            try:
                assert (self.truth_test_matrix + self.truth_train_matrix != 2).all()
            except AssertionError as e:
                print("Overlap in test and training truth matrices in {} cases".format(np.sum(self.truth_test_matrix + self.truth_train_matrix == 2)))
                raise e
        if self.truth_val_matrix is not None and self.truth_train_matrix is not None:
            try:
                assert (self.truth_val_matrix + self.truth_train_matrix != 2).all()
            except AssertionError as e:
                print("Overlap in validation and training truth matrices in {} cases".format(np.sum(self.truth_val_matrix + self.truth_train_matrix == 2)))
                raise e
        if self.truth_test_matrix is not None and self.truth_val_matrix is not None:
            try:
                assert (self.truth_test_matrix + self.truth_val_matrix != 2).all()
            except AssertionError as e:
                print("Overlap in test and validation truth matrices in {} cases".format(np.sum(self.truth_test_matrix + self.truth_val_matrix == 2)))
                raise e

    def evaluate(self, similarity, use_testing: bool = False, both_directions: bool = False): 
        ''' Evaluates similarity matrix in respect to the truth matrices
            similarity matric must be in the format that a high value at similarity[c,d] indicates a high
            probability of an edge betweend c and d.
            If use_testing is false it calculates mrr for training and validation and appends the result to
            self.mrrs_x_y with x being [row,col] and y being [train, val]. If use_testing is true, it calculates mrr for the
            test data, creates the lists self.mrrs_x_test with x being [row,col] and places the result there.'''

        ind = np.argsort(similarity, axis=1)

        # from row to column

        ordered_truth_val = np.take_along_axis(self.truth_val_matrix, ind, axis=1)
        ordered_truth_train = np.take_along_axis(self.truth_train_matrix, ind, axis=1)

        if use_testing:
            ordered_truth_test = np.take_along_axis(self.truth_test_matrix, ind, axis=1)
            mrr_row_test = self.mean_reciprocal_rank(ordered_truth_test,ordered_truth_train+ordered_truth_val)
        
        mrr_row_val = self.mean_reciprocal_rank(ordered_truth_val,ordered_truth_train)
        mrr_row_train = self.mean_reciprocal_rank(ordered_truth_train,ordered_truth_val)

        ind_transpose = np.argsort(similarity.transpose(), axis=1)

        #from column to row

        ordered_truth_val_transpose = np.take_along_axis(self.truth_val_matrix.transpose(), ind_transpose, axis=1)
        ordered_truth_train_transpose = np.take_along_axis(self.truth_train_matrix.transpose(), ind_transpose, axis=1)

        if use_testing:
            ordered_truth_test_transpose = np.take_along_axis(self.truth_test_matrix.transpose(), ind_transpose, axis=1)
            mrr_col_test = self.mean_reciprocal_rank(ordered_truth_test_transpose,ordered_truth_train_transpose+ordered_truth_val_transpose)
        
        mrr_col_val = self.mean_reciprocal_rank(ordered_truth_val_transpose,ordered_truth_train_transpose)
        mrr_col_train = self.mean_reciprocal_rank(ordered_truth_train_transpose,ordered_truth_val_transpose)

        if both_directions:
            # experimental feature, seems like it yields the same result as just averaging the col and row mrr
            list0_train = [rs for rs in ordered_truth_train]
            list1_train = [rs for rs in ordered_truth_train_transpose]
            both_train = list0_train + list1_train

            list0_val = [rs for rs in ordered_truth_val]
            list1_val = [rs for rs in ordered_truth_val_transpose]
            both_val = list0_val + list1_val
            #ordered_truth_val_transpose = np.concatenate((np.zeros((1367,ordered_truth_val_transpose.shape[1])),ordered_truth_val_transpose),axis=0)

            #ordered_truth_train_both = np.concatenate((ordered_truth_train, ordered_truth_train_transpose),1)
            #ordered_truth_val_both = np.concatenate((ordered_truth_val, ordered_truth_val_transpose),1)
            #print(ordered_truth_train_both.shape)
            
            '''
            mrr_both_train = self.mean_reciprocal_rank([rs for rs in ], ordered_truth_val_both)
            mrr_both_val = self.mean_reciprocal_rank(ordered_truth_val_both, ordered_truth_train_both)

            if use_testing:
                ordered_truth_test_transpose = np.concatenate((np.zeros((1367,ordered_truth_test_transpose.shape[1])),ordered_truth_test_transpose),axis=0)
                ordered_truth_test_both = np.concatenate((ordered_truth_test, ordered_truth_test_transpose),1)
                mrr_both_test = self.mean_reciprocal_rank(ordered_truth_test_both,ordered_truth_train_both+ordered_truth_val_both)
                self.mrrs_both_test = [mrr_both_test[1]]

            self.mrrs_both_train = [mrr_both_train[1]]
            self.mrrs_both_val = [mrr_both_val[1]]
            '''


        if use_testing:
            self.mrrs_row_test = [mrr_row_test[1]]
            self.mrrs_col_test = [mrr_col_test[1]]   
            self.mean_ranks_row_test = [mrr_row_test[3]]
            self.mean_ranks_col_test = [mrr_col_test[3]]   
            self.hat5_row_test = [mrr_row_test[5]]
            self.hat5_col_test = [mrr_col_test[5]]  
            self.hat10_row_test = [mrr_row_test[7]]
            self.hat10_col_test = [mrr_col_test[7]]  
            self.hat20_row_test = [mrr_row_test[9]]
            self.hat20_col_test = [mrr_col_test[9]]  
            self.hat50_row_test = [mrr_row_test[11]]
            self.hat50_col_test = [mrr_col_test[11]]  
               
        
        self.mrrs_row_train.append(mrr_row_train[1])
        self.mrrs_row_val.append(mrr_row_val[1])
        self.mrrs_col_train.append(mrr_col_train[1])
        self.mrrs_col_val.append(mrr_col_val[1])

        self.mean_ranks_row_train.append(mrr_row_train[3])
        self.mean_ranks_row_val.append(mrr_row_val[3])
        self.mean_ranks_col_train.append(mrr_col_train[3])
        self.mean_ranks_col_val.append(mrr_col_val[3])

        self.hat5_row_train.append(mrr_row_train[5])
        self.hat5_row_val.append(mrr_row_val[5])
        self.hat5_col_train.append(mrr_col_train[5])
        self.hat5_col_val.append(mrr_col_val[5])

        self.hat10_row_train.append(mrr_row_train[7])
        self.hat10_row_val.append(mrr_row_val[7])
        self.hat10_col_train.append(mrr_col_train[7])
        self.hat10_col_val.append(mrr_col_val[7])

        self.hat20_row_train.append(mrr_row_train[9])
        self.hat20_row_val.append(mrr_row_val[9])
        self.hat20_col_train.append(mrr_col_train[9])
        self.hat20_col_val.append(mrr_col_val[9])

        self.hat50_row_train.append(mrr_row_train[11])
        self.hat50_row_val.append(mrr_row_val[11])
        self.hat50_col_train.append(mrr_col_train[11])
        self.hat50_col_val.append(mrr_col_val[11])

    def random(self, n_times=10, seed=None, proximity=None, use_testing=False):

        if seed != None:
            np.random.seed(seed)

        if proximity is not None:
            for i in range(n_times):
                np.random.shuffle(proximity)
                self.evaluate(proximity, use_testing)
                print(".",end="")
            
        else:
            for i in range(n_times):
                proximity = np.random.rand(self.truth_train_matrix.shape[0],self.truth_train_matrix.shape[1])
                self.evaluate(proximity, use_testing)
                print(".",end="")
        print("")
        
        print(np.mean(self.mrrs_row_train[-n_times:]))
        print(np.mean(self.mrrs_row_val[-n_times:]))
        if use_testing:
            print(np.mean(self.mrrs_row_test[-n_times:]))
        print(np.mean(self.mrrs_col_train[-n_times:]))
        print(np.mean(self.mrrs_col_val[-n_times:]))
        if use_testing:
            print(np.mean(self.mrrs_col_test[-n_times:]))

        self.mrrs_row_train = self.mrrs_row_train[:-n_times]
        self.mrrs_row_val = self.mrrs_row_val[:-n_times]
        self.mrrs_col_train = self.mrrs_col_train[:-n_times]
        self.mrrs_col_val = self.mrrs_col_val[:-n_times]

    def mean_reciprocal_rank(self, rs, additional_truth=[None], get_hits=[]):
        """Score is reciprocal of the rank of the first relevant item
        First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
        Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
        >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        >>> mean_reciprocal_rank(rs)
        0.61111111111111105
        >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        >>> mean_reciprocal_rank(rs)
        0.5
        >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
        >>> mean_reciprocal_rank(rs)
        0.75
        Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean reciprocal rank
        """
        # NOTE: array is reversed during the procedure, since np.argsort only allows for sorting in ascending order
        rs_raw = list(np.asarray(r[::-1]).nonzero()[0] for r in rs if np.sum(r) > 0)
        
        mrr_raw = np.mean([1. / (r + 1) for sublist in rs_raw for r in sublist]) 
        mean_rank_raw = np.mean([r + 1 for sublist in rs_raw for r in sublist]) 

        hitsat5_raw = np.mean([1 if r < 5 else 0 for sublist in rs_raw for r in sublist]) 
        hitsat10_raw = np.mean([1 if r < 10 else 0 for sublist in rs_raw for r in sublist]) 
        hitsat20_raw = np.mean([1 if r < 20 else 0 for sublist in rs_raw for r in sublist]) 
        hitsat50_raw = np.mean([1 if r < 50 else 0 for sublist in rs_raw for r in sublist]) 

        total_before = np.sum(rs)

        rs_filtered = []
        if (additional_truth != None).all():
        #if True:
            # also remove all known true examples from the other sets
            rs_prefiltered = []
            for i, additional in enumerate(additional_truth):
                to_delete = additional.nonzero()[0]
                rs_prefiltered.append(np.delete(rs[i],to_delete)) 
            
            total_after = np.sum(np.sum(r) for r in rs_prefiltered)
            assert  total_before == total_after # nothing lost filtering for out-of-sample edges
            rs = rs_prefiltered

        for r in rs:
            while np.sum(r) > 0:
                best = r.nonzero()[0][-1]
                best_r = np.zeros_like(r)
                best_r[best] = 1
                rs_filtered.append(best_r)
                r = np.delete(r, best)

        total_after = np.sum(np.sum(r) for r in rs_filtered)
        assert total_before == total_after # nothing lost in filtering
        assert len(rs_filtered) == total_before # every edge gets its own array
        for r in rs_filtered:
            assert np.sum(r) == 1 # only one edge in every array

        # NOTE: array is reversed during the procedure, since np.argsort only allows for sorting in ascending order
        rs_filtered = list(np.asarray(r[::-1]).nonzero()[0] for r in rs_filtered if np.sum(r) > 0)
        mrr_filtered = np.mean([1. / (r + 1) if r.size else 0. for r in rs_filtered]) 
        mean_rank_filtered = np.mean([r + 1 if r.size else 0. for r in rs_filtered]) 

        hitsat5_filtered = np.mean([1 if r < 5 else 0 for sublist in rs_filtered for r in sublist])
        hitsat10_filtered = np.mean([1 if r < 10 else 0 for sublist in rs_filtered for r in sublist])
        hitsat20_filtered = np.mean([1 if r < 20 else 0 for sublist in rs_filtered for r in sublist]) 
        hitsat50_filtered = np.mean([1 if r < 50 else 0 for sublist in rs_filtered for r in sublist])

        return (mrr_raw, mrr_filtered, mean_rank_raw, mean_rank_filtered, hitsat5_raw, hitsat5_filtered, hitsat10_raw, hitsat10_filtered,
            hitsat20_raw, hitsat20_filtered, hitsat50_raw, hitsat50_filtered)


    

def main():
    similarity = np.genfromtxt("misc/word2vecresults/similarity_matrix_fold1.csv", delimiter = ",")[1:,1:]
    truth_test = np.genfromtxt("misc/word2vecresults/truth_matrix_test_fold1.csv", delimiter = ",")[1:,1:]
    truth_train = np.genfromtxt("misc/word2vecresults/truth_matrix_train_fold1.csv", delimiter = ",")[1:,1:]

    evaluator = Evaluator()
    evaluator.truth_val_matrix = truth_test
    evaluator.truth_train_matrix = truth_train
    evaluator.evaluate(similarity)
    
if __name__ == '__main__':
    main()