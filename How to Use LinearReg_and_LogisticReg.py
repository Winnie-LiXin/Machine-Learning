import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("path_to_file")  ## e.g. sys.path.append(r"C:\Users\Winnie Lee\Desktop\")
from LinearReg_and_LogisticReg import *

if __name__ == "__main__":

	print("******************************")
	print("*   Regression Start    *")
	print("******************************")
	linear_model(train_file='TrainData_LinearReg.csv', 
              test_file='TestData_LinearReg.csv', 
              num_iterations=6000, learning_rate=0.01, r_type='linear')
	print("\n")
	logistic_model(train_file='TrainData_LogisticReg.csv', 
                test_file='TestData_LogisticReg.csv', 
                num_iterations=10000, learning_rate=0.03, r_type='logistic')
