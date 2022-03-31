import numpy as np
import pandas as pd
import joblib
from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder



### Train a model to work out the compressive strength based on the materials.
### Use EDA to find what may be the most significant materials to change.
### Run a selection of untested combinations through the model.
### Output the suspected compression strength.
### Find a combination that is higher than the current compression strength.

class ConcreteRegressor:
    def __init__(self, data_path):
        self.base_data = data_path
        self.clean_data = None
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.best_score = None
        self.best_params = None

    def prepare_data(self, data=None):
        """
        data_path should be a string path to where the data is stored. We expect a CSV with 9 columns,
        the first 8 are materials that concrete is comprised of. The 9th column is the compression strength of that column.
        """

        # Check we aren't loading data with this function and on Regressor Creation
        df = pd.read_csv(self.base_data)
        if df.shape[1] == 9 :
            df.columns = ['cement', 'BG Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Compressive Strength']
            self.clean_data = df
            return "success"
        else:
            print(f"We expected to build a dataframe of shape (*, 9), instead we built {df.shape}, check your data.csv")
            return "failure"

    def load_and_split_data(self, test_size=0.2):
        """
        data: Should be a string path to a CSV with a label field of 'Compressive Strength'.
        test_size: the percent of data to save for testing, default 20%

        This function will take the path to a CSV. Split the data into train and tests and set
        the instances variables.
        """
        if isinstance(self.clean_data, pd.DataFrame):
            X = self.clean_data.drop('Compressive Strength', axis=1)
            Y = self.clean_data['Compressive Strength']
            self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(X, Y, test_size=test_size)

            #print(f"{self.train_data.shape}\n{self.train_label.shape}\n{self.test_data.shape}\n{self.test_label.shape}")

        else:
            if self.base_data:
                print("Data needs to be prepared before we split it, run ConcreteRegressor.clean_data to before you continue")
            else:
                print("""
                No data has beenn loaded into the ConcreteRegressor yet, you can add it using the 'prepare_data' function
                with an argument of a string path to the CSV containing the dataset
                """)

    def find_best_training_params(self):
        print("ENTERING TRAINING MODE TO FIND BEST PARAMETERS....")
        print("This may take a while")

        criterion_options = {'squared_error', 'absolute_error'}
        max_depth = 2
        min_samples_split = 2
        min_sample_leaf = 2
        total_count = 0
        for criterion in criterion_options:
            for max_depth_val in range(1, max_depth + 1):
                for min_samples_split_val in range(2, min_samples_split + 1 ):
                    for min_sample_leaf_val in range(1, min_sample_leaf + 1 ):
                        total_count +=1
                        current_params = {
                                            "criterion" : criterion,
                                            "max_depth_val": max_depth_val,
                                            "min_samples_split_val": min_samples_split_val,
                                            "min_samples_leaf_val": min_sample_leaf_val,
                                            }
                        regressor = RandomForestRegressor(  criterion = criterion,
                                                            max_depth = max_depth_val,
                                                            min_samples_split = min_samples_split_val,
                                                            min_samples_leaf = min_sample_leaf_val,
                                                         )
                        regressor.fit(self.train_data, self.train_label)
                        test_score = regressor.score(self.test_data, self.test_label)

                        if self.best_score and self.best_score < test_score:
                            print(f"Found a new best...{test_score}: \n {current_params}")

                            self.best_score = test_score
                            self.best_params = current_params
                        else:
                            if not self.best_score:
                                self.best_score = test_score
                                self.best_params = current_params

                        ## Updates the user on how far we are through testing diferent parameters.
                        if total_count % 40 == 0:
                            print(f"testing parameter combinationn {total_count} of 840")
                            print(current_params)

    def run_production_training(self):
        self.prepare_data()
        if self.best_params:
            train_data = self.clean_data.drop('Compressive Strength', axis=1)
            train_label = self.clean_data['Compressive Strength']
            regressor = RandomForestRegressor(  criterion =  self.best_params['criterion'],
                                                max_depth = self.best_params['max_depth_val'],
                                                min_samples_split = self.best_params['min_samples_split_val'],
                                                min_samples_leaf = self.best_params['min_samples_leaf_val'] )
            regressor.fit(train_data, train_label)
            joblib.dump(regressor, "data/saved_models/production_forest.joblib")
        else:
            print("We need to find the best parameters first. Try using .find_best_training_params() before running production training")

    def production_predict(self, concrete_materials):
        """
        Concrete_materials should be a dictionary, the keys should be the same as the training column names.
        """
        ingredient_df = pd.DataFrame.from_dict(conrete_materials)
        regressor = joblib.load("data/saved_models/production_forest.joblib")
