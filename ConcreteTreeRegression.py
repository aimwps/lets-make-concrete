from models import ConcreteRegressor
import sys
from os.path import exists
import joblib
import pandas as pd
import datetime as dt
MENU_OPTIONS = {
                "1": "Find best parameters",
                "2": "Train production model with best parameters",
                "3": "Manually enter a singular concrete composition",
                "4": "Enter a path to a CSV containing multiple compositions",
                "Q": "To quit"
}

entry = {   'cement': [],
            'BG Slag': [],
            'Fly Ash': [],
            'Water': [],
            'Superplasticizer':[],
            'Coarse Aggregate': [],
            'Fine Aggregate': [],
            'Age': [],
        }


def print_menu_options():
    for key_code, action in MENU_OPTIONS.items():
        print(f"Hit {key_code}: {action}")

if __name__ == "__main__":
    finished = False
    regressor = ConcreteRegressor("data/Concrete_Data.csv")
    print("\nWelcome to the concrete mixer")
    print("We predict compression strengths of concrete based on the composition of ingredients")
    print("Select an option to continue:\n")
    while not finished:
        print_menu_options()
        user_input = input("\nEnter an option followed by the Enter key: ")
        while user_input not in MENU_OPTIONS.keys():
            print(ord(user_input), type(ord(user_input)))
            user_input = input("That is not a valid option, try again: ")

        if user_input == "1":
            if not regressor.clean_data:
                regressor.prepare_data()
            if not regressor.train_data or not regressor.train_label or not regressor.test_data or not regressor.test_label:
                regressor.load_and_split_data() # you can enter a test_size=<float> here to change test size.

            regressor.find_best_training_params()
            print("\n***Sucessfully found the best parameters: ")
            print(f"{regressor.best_params} ***\n")


        elif user_input =="2":
            if regressor.best_params:
                regressor.run_production_training()
                print("\n***Success we've completed running production training***\n")
            else:
                print("You need to find the best parameters (option 1) before training a production model")

        elif user_input =="3":
            if exists("data/saved_models/production_forest.joblib"):
                production_regressor = joblib.load("data/saved_models/production_forest.joblib")
                new_entry = entry
                for key in new_entry.keys():
                    data_entry_valid = False
                    while not data_entry_valid:
                        data_entry = input(f"{key}: ")
                        try:
                            data_entry = float(data_entry)
                            new_entry[key].append(data_entry)
                            data_entry_valid = True
                        except:
                            data_entry_valid = False

                predict_df = pd.DataFrame(new_entry)
                result = production_regressor.predict(predict_df)
                print(f"\n ***These values give a compression rating of:  {result}*** \n")

                result_df = predict_df.copy()
                result_df['Compressive Strength'] = result

                sorted_df = result_df.sort_values(by=['Compressive Strength'], ascending=False )
                date_string = dt.datetime.now().strftime("%d%m%y_%H%M%S%f")
                filename = f"data/results/{date_string}_results.csv"
                sorted_df.to_csv(filename)

            else:
                print("There is not a trained production model saved, you will need to run options 1 and 2 first")

        elif user_input =="4":
            if exists("data/saved_models/production_forest.joblib"):
                valid_path_to_file = False
                while not valid_path_to_file:
                    print("Enter a relative path to concrete composition data (must be of type .csv)")
                    user_file_input = input("path: ")
                    if exists(user_file_input):
                        valid_path_to_file = True
                    else:
                        print("We couldn't find any data at that path, try again.\n")
            else:
                print("There is not a trained production model saved, you will need to run options 1 and 2 first")
        else:
            if user_input == "Q" or user_input == "q":
                finished = True
                sys.exit()
