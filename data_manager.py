import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''This class helps to create te desired synthetic dataset for experimentations'''


class DatasetManager:

    def __init__(self, n_examples, n_features, version, factor=None):
        self.n_examples = n_examples
        self.n_features = n_features
        self.y = None
        self.version = version
        self.data = None
        self.noise = None
        '''For outliers'''
        self.factor = factor
        print("Options: xor, outliers, circular, regression, regression_square, additive_non_linear,\nadditive_non_linear_with_product+2+3, mix_circular_additive, F1, F2, F3, F4, F5\n\nCorrelation: simple, non_linear, full")

    def xor_data(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = -np.sign(self.data[:, 0]*self.data[:, 1])
        y[y < 0] = 0
        self.y = y.reshape(-1, 1)

    def regression_square_data(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (self.data[:, 0]*self.data[:, 1]
             + 0.5*np.square(self.data[:, 2])
             + 1
             ).reshape(-1, 1)
        self.y = y

    def regression_data(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (4*self.data[:, 0]
             - 4
             + self.noise
             ).reshape(-1, 1)
        self.y = y

    def outlier_data(self):
        assert self.factor != None, "ERROR: Must provide an outlier factor coefficient!"

        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 0.1, size=(self.n_examples))
        y = (4*self.data[:, 0]
             - 4
             + self.noise
             ).reshape(-1, 1)
        ind = [x for x in range(self.n_examples)]
        index_outliers = np.random.choice(ind, size=20, replace=False)
        for i in index_outliers:
            p = np.random.random()
            if p > 0.5:
                self.data[i, 0] = self.data[i, 0] + \
                    np.std(self.data[:, 0])*self.factor
            else:
                self.data[i, 0] = self.data[i, 0] - \
                    np.std(self.data[:, 0])*self.factor
        self.y = y

    def circular_data(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = np.round(1/(1+np.exp(
            4*np.square(self.data[:, 0])
            - 2*np.square(self.data[:, 1])
            + np.square(self.data[:, 2])
            - 3*np.square(self.data[:, 3])
            
        )
        )
        ).reshape(-1, 1)
        self.y = y
    
    def circular_data_reg(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (
            4*np.square(self.data[:, 0])
            - 2*np.square(self.data[:, 1])
            + np.square(self.data[:, 2])
            - 3*np.square(self.data[:, 3])
            ).reshape(-1, 1)
        self.y = y

    def additive_non_linear(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = np.round(1/(1+np.exp((-10*np.sin(2*self.data[:, 0])
                                  + 2*np.abs(self.data[:, 1])
                                  + self.data[:, 2]
                                  - np.exp(-self.data[:, 3])
                                  )
                                 )
                        )).reshape(-1, 1)
        self.y = y
        
    def additive_non_linear_reg(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (-10*np.sin(2*self.data[:, 0])
                                  + 2*np.abs(self.data[:, 1])
                                  + self.data[:, 2]
                                  - np.exp(-self.data[:, 3])
            ).reshape(-1, 1)
        self.y = y

    def additive_non_linear_with_product(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = np.round(1/(1+np.exp(-(4*self.data[:, 0]*self.data[:, 1]*self.data[:, 2]
             + self.data[:, 3]*self.data[:, 4]*self.data[:, 5]
             )))).reshape(-1, 1)

        self.y = y
        
    def additive_non_linear_with_product_reg(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (4*self.data[:, 0]*self.data[:, 1]*self.data[:, 2]
             + self.data[:, 3]*self.data[:, 4]*self.data[:, 5]
             ).reshape(-1, 1)

        self.y = y
    def additive_non_linear_with_product2(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = np.round(1/(1+np.exp(-(10*np.sin(self.data[:, 0]*self.data[:, 1]*self.data[:, 2])
             + np.abs(self.data[:, 3]*self.data[:, 4]*self.data[:, 5])
             )))).reshape(-1, 1)

        self.y = y
        
    def additive_non_linear_with_product2_reg(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (10*np.sin(self.data[:, 0]*self.data[:, 1]*self.data[:, 2])
             + np.abs(self.data[:, 3]*self.data[:, 4]*self.data[:, 5])
             ).reshape(-1, 1)

        self.y = y
        
    def additive_non_linear_with_product3(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = np.round(1/(1+np.exp((-20*np.sin(2*self.data[:, 0]*self.data[:, 1])
                                  + 2*np.abs(self.data[:, 2])
                                  + self.data[:, 3]*self.data[:, 4]
                                  - 4*np.exp(-self.data[:, 5])
                                  )
                                 )
                        )).reshape(-1, 1)
        self.y = y
        
    def additive_non_linear_with_product3_reg(self):
        self.data = np.random.normal(
            0, 1, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 1, size=(self.n_examples))
        y = (-20*np.sin(2*self.data[:, 0]*self.data[:, 1])
                                  + 2*np.abs(self.data[:, 2])
                                  + self.data[:, 3]*self.data[:, 4]
                                  - 4*np.exp(-self.data[:, 5])
            ).reshape(-1, 1)
        self.y = y

    def F1(self):
        self.data = np.random.uniform(
            0, 1, size=(self.n_examples, self.n_features))
        self.data[:, 3] = np.random.uniform(0.6, 1)
        self.data[:, 4] = np.random.uniform(0.6, 1)
        self.data[:, 7] = np.random.uniform(0.6, 1)
        self.data[:, 9] = np.random.uniform(0.6, 1)
        y = ((np.pi**(self.data[:, 0]*self.data[:, 1]))*np.sqrt(self.data[:, 2])
             + 1/np.sin(self.data[:, 3]) +
             np.log(self.data[:, 2]+self.data[:, 4])
             - (self.data[:, 8]/self.data[:, 9]) *
             np.sqrt(self.data[:, 6]/self.data[:, 7])
             - self.data[:, 1]*self.data[:, 6]
             ).reshape(-1, 1)
        self.y = y

    def F2(self):
        self.data = np.random.uniform(-1, 1,
                                      size=(self.n_examples, self.n_features))
        y = (np.exp(np.abs(self.data[:, 0]-self.data[:, 1]))
             + np.abs(self.data[:, 1]*self.data[:, 2])
             - np.square(self.data[:, 2])**np.abs(self.data[:, 3])
             + np.square(self.data[:, 0]*self.data[:, 3])
             + np.log(np.square(self.data[:, 3]) +np.square(self.data[:, 4]) +
                      np.square(self.data[:, 6])+np.square(self.data[:, 7]))
             + self.data[:, 8]
             + 1/(1+np.square(self.data[:, 9]))
             ).reshape(-1, 1)
        self.y = y
        
    def F3(self):
        self.data = np.random.uniform(-1, 1,
                                      size=(self.n_examples, self.n_features))
        y = (np.sin(np.abs(self.data[:, 0]*self.data[:, 1])+1)
             - np.log(np.abs(self.data[:, 2]*self.data[:, 3])+1)
             + np.cos(self.data[:, 4]+self.data[:, 5]-self.data[:, 7])
             + np.sqrt(np.square(self.data[:, 7]) +np.square(self.data[:, 8]) +
                      np.square(self.data[:, 9]))
        
             ).reshape(-1, 1)
        self.y = y
        
    def F4(self):
        self.data = np.random.uniform(-1, 1,
                                      size=(self.n_examples, self.n_features))
        y = ( np.tanh(self.data[:, 0]*self.data[:, 1]+self.data[:, 2]*self.data[:, 3])*np.sqrt(np.abs(self.data[:, 4])) 
             + np.log((self.data[:, 5]*self.data[:, 6]*self.data[:, 7])**2 + 1) 
             + self.data[:, 8]*self.data[:, 9] 
             + 1./(1+np.abs(self.data[:, 9]))
             ).reshape(-1, 1)
        self.y = y
        
    def F5(self):
        self.data = np.random.uniform(-1, 1,
                                      size=(self.n_examples, self.n_features))
        y = (np.cos(self.data[:, 0]*self.data[:, 1]*self.data[:, 2])
             + np.sin(self.data[:, 3]*self.data[:, 4]*self.data[:, 5])  
             ).reshape(-1, 1)
        self.y = y

    def mix_circular_additive(self):
        self.data = np.random.normal(
            0, 2, size=(self.n_examples, self.n_features))
        self.noise = np.random.normal(0, 4, size=(self.n_examples))
        y1 = np.round(1/(1+np.exp(
            np.square(self.data[:self.n_examples//2, 5])
            + np.square(self.data[:self.n_examples//2, 6])
            + np.square(self.data[:self.n_examples//2, 7])
            + np.square(self.data[:self.n_examples//2, 8])
            - 4
        )
        )
        ).reshape(-1, 1)

        y2 = np.round(1/(1+np.exp(
            -100*np.sin(2*self.data[self.n_examples//2:, 5])
            + 2*np.abs(self.data[self.n_examples//2:, 6])
            + self.data[self.n_examples//2:, 7]
            + np.exp(-self.data[self.n_examples//2:, 8])
        )
        )
        ).reshape(-1, 1)

        y_loc = np.concatenate([y1, y2], axis=0)
        self.y = y_loc

    def correlation_simple(self):
        self.data[:, 0] = 3*self.data[:, 1]+9

    def correlation_non_linear(self):
        self.data[:, 9] = np.sin(np.exp(self.data[:, 7]))

    def correlation_both(self):
        self.data[:, 9] = np.sin(np.exp(self.data[:, 7]))
        self.data[:, 3] = 2*self.data[:, 5]+0.5

    def plot_distribution(self):
        fig = plt.figure(figsize=(5, 5))
        plt.hist(self.y)

    def plot_outliers(self):
        fig = plt.figure(figsize=(5, 5))
        plt.boxplot(self.data[:, 0])

    def generate_dataset(self, correlation=None, display=True):
        '''Datasets'''
        if self.version is "xor":
            self.xor_data()
        elif self.version is "outlier":
            self.outlier_data()
        elif self.version is "circular":
            self.circular_data()
        elif self.version is "circular_reg":
            self.circular_data_reg()
        elif self.version is "regression":
            self.regression_data()
        elif self.version is "regression_square":
            self.regression_square_data()
        elif self.version is "additive_non_linear":
            self.additive_non_linear()
        elif self.version is "additive_non_linear_reg":
            self.additive_non_linear_reg()
        elif self.version is "additive_non_linear_with_product":
            self.additive_non_linear_with_product()
        elif self.version is "additive_non_linear_with_product2":
            self.additive_non_linear_with_product2()
        elif self.version is "additive_non_linear_with_product3":
            self.additive_non_linear_with_product3()
        elif self.version is "additive_non_linear_with_product_reg":
            self.additive_non_linear_with_product_reg()
        elif self.version is "additive_non_linear_with_product2_reg":
            self.additive_non_linear_with_product2_reg()
        elif self.version is "additive_non_linear_with_product3_reg":
            self.additive_non_linear_with_product3_reg()
        elif self.version is "mix_circular_additive":
            self.mix_circular_additive()
        elif self.version is "F1":
            self.F1()
        elif self.version is "F2":
            self.F2()
        elif self.version is "F3":
            self.F3()
        elif self.version is "F4":
            self.F4()
        elif self.version is "F5":
            self.F5()
        '''Correlations'''
        if correlation is "simple":
            self.correlation_simple()
        elif correlation is "non_linear":
            self.correlation_non_linear()
        elif correlation is "full":
            self.correlation_both()
        assert (self.y is not None), "Please provide a correct distribution"
        if display is True:
            self.plot_distribution()
        if self.version is "outlier":
            self.plot_outliers()
        return self.y, self.data
