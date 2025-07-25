'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    etl.run_etl()

    preprocessing.preprocess_data()

    logistic_regression.run_logistic()

    decision_tree.run_decision_tree()

    calibration_plot.run_calibration()

    if __name__ == "__main__":
        main()