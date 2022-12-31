'''
salary_predictor.py
Predictor of salary from old census data -- riveting!
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *

class SalaryPredictor:

    def __init__(self, X_train, y_train):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Performs and fits
        any preprocessing methods (e.g., imputing of missing features,
        discretization of continuous variables, etc.) on the inputs, and saves
        these as attributes to later transform test inputs.
        
        :param DataFrame X_train: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        columns_to_clean = X_train.columns 
        self.imputer = SimpleImputer(missing_values = '?', strategy ='most_frequent') 
        self.imputer.fit(X_train)
        clean_features = self.clean_data_helper(X_train, columns_to_clean) 

        constant_cols = clean_features.filter(items = ['age','education_years','capital_gain','class'])
        changing_cols = clean_features.filter(items = ['education', 'marital', 'occupation_code','country'])

        self.one_hot = OneHotEncoder(handle_unknown='ignore')
        self.one_hot.fit(changing_cols)
        one_hot_features = self.one_hot.transform(changing_cols).toarray()
        concatenated_cols = pd.concat((pd.DataFrame(one_hot_features), pd.DataFrame(constant_cols.to_numpy())), axis=1)
        self.model = LogisticRegression( max_iter = 35881).fit(concatenated_cols, y_train)



        
    def classify (self, X_test):
        """
        Takes a DataFrame of rows of input attributes of census demographic
        and provides a classification for each. Note: must perform the same
        data transformations on these test rows as was done during training!
        
        :param DataFrame X_test: DataFrame of rows consisting of demographic
        attributes to be classified
        :return: A list of classifications, one for each input row X=x
        """
        good_cols = X_test.columns
        clean_cols = self.clean_data_helper(X_test, good_cols) 
        constant_cols = clean_cols.filter(items = ['age', 'education_years','capital_gain','class'])
        changing_cols = clean_cols.filter(items = ['education','marital','occupation_code','country'])
        one_hot_features = self.one_hot.transform(changing_cols).toarray()
        
        feats = pd.concat((pd.DataFrame(one_hot_features), pd.DataFrame(constant_cols.to_numpy())), axis=1)
        return self.model.predict(feats)
    
    def test_model (self, X_test, y_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test demographics
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame X_test: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_test: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        classified = self.classify(X_test)
        print(str(classification_report(y_test, classified)))
    

    def clean_data_helper(self, frame, columns_to_clean): 
        good_cols = frame.filter(items = columns_to_clean) 
        features = pd.DataFrame(self.imputer.transform(good_cols), columns = good_cols.columns)
        features.index = good_cols.index
        return features

def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with the sanitized
    data (e.g., removing leading / trailing spaces).
    NOTE: This should *not* do the preprocessing like turning continuous
    variables into discrete ones, or performing imputation -- those
    functions are handled in the SalaryPredictor constructor, and are
    used to preprocess all incoming test data as well.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the demographic
    information and labels. It is assumed that for n columns, the first
    n-1 are the inputs X and the nth column are the labels y
    """
    data = pd.read_csv(data_file)
    print(data.head())
    return data


if __name__ == "__main__":
    data = load_and_sanitize("./dat/salary.csv")
    X_train, X_test, y_train, y_test = train_test_split(data.drop(axis = 1, columns = ['class']), data["class"]) 
    datay = SalaryPredictor(X_train= X_train, y_train= y_train)
    print("-- classification report w/ Y_test: " )
    print(str(classification_report(y_test, datay.classify(X_test))))

    print("+++++++++++++++++ Testing the Model+++++++++++++++++")
    datay.test_model(X_test, y_test)



    
    