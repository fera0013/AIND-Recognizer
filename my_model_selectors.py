import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def BIC(self, model , num_states):
        logL = model.score(self.X, self.lengths)
        parameters = num_states * num_states + 2 * num_states * len(self.X[0]) - 1 
        return (-2) * logL + math.log(len(self.X)) * parameters
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        lowest_BIC = float('inf')
        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                model = self.base_model(num_states)
                bic = self.BIC(model, num_states)
                if bic < lowest_BIC:
                    best_model = model
                    lowest_BIC = bic
            except: 
                continue
        return model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection based on DIC scores
        # for each of the num hidden components to try
        # fit on train
        # score on train
        # score on everything else
        # dhfhdf
        results={}
        antiRes={}
        DICs=[]
        dic = []
        try:
            hmm_model=None
            for num_states in range(self.min_n_components,self.max_n_components+1):
                antiLogL = 0.0
                wc = 0
                hmm_model =  self.base_model(num_states)
                logL = hmm_model.score(self.X, self.lengths)             
                for word in self.hwords:
                    if word == self.this_word:
                        continue
                    X, lengths = self.hwords[word]
                    antiLogL += hmm_model.score(X, lengths)
                    wc += 1
                antiLogL /= float(wc)
                results[num_states] = logL
                antiRes[num_states] = antiLogL
                dic = results[num_states] -  antiRes[num_states]
                DICs.append((hmm_model,dic))
        except:
            pass
        if dic == []:
               return None
        return max(DICs, key= lambda x: x[1])[0]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        best_score = float('-inf')
        best_model = GaussianHMM()
        split_method = KFold(min(3,len(self.sequences)))
        for num_states in range(self.min_n_components,self.max_n_components+1):
            for train_idx, test_idx in split_method.split(self.sequences):
                X_train , train_length = combine_sequences(train_idx, self.sequences)
                self.X = X_train
                self.lengths =  train_length
                try:
                    model = self.base_model(num_states)
                    X_test , test_length = combine_sequences(test_idx, self.sequences)
                    logL = model.score( X_test, test_length)
                    if logL > best_score:
                        best_score = logL
                        best_model = model
                except:
                    continue
        return best_model
