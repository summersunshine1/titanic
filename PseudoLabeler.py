from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np
import random

class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    '''
    
    def __init__(self, model, unlabled_data, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
        
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.unlabled_data = unlabled_data
        # self.features = features
        # self.target = target
        
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            # "features": self.features,
            # "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''

        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[:,:-1],
            augemented_train[:,-1]
        )
        
        return self


    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''        
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data)
        
        # Add the pseudo-labels to the test set
        length = len(self.unlabled_data)
        indexs = list(range(0,length))
        random.seed(a=19)
        sampleindex = random.sample(indexs,num_of_samples)

        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        pseudo_labels = np.array([[p] for p in pseudo_labels])
        sampled_pseudo_data = np.hstack((self.unlabled_data[sampleindex], pseudo_labels[sampleindex]))
        y1 = np.array([[t] for t in y])
        temp_train = np.hstack((X, y1))
       
        augemented_train = np.vstack((sampled_pseudo_data, temp_train))
        augemented_train = np.array(augemented_train)
        return shuffle(augemented_train)
        
    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)
    
    def get_model_name(self):
        return self.model.__class__.__name__