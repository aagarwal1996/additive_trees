import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


'''
This python script is used to compute predictions and MSE for honest trees and forests. 

'''

def get_honest_leaf_averages(CART,X_honest,y_honest):
    X_honest_leaf_ids = CART.apply(X_honest)
    unique_leaf_ids = np.unique(X_honest_leaf_ids)
    X_honest_leaf_node_ids = {k: v for k, v in enumerate(X_honest_leaf_ids)}
    leaf_id_to_honest_av = {}
    for leaf_id in unique_leaf_ids:
        leaf_id_to_honest_av[leaf_id] = y_honest[[k for k,v in X_honest_leaf_node_ids.items() if v == leaf_id]].mean()
    return leaf_id_to_honest_av

def get_honest_tree_test_preds(CART,X_test,y_test,leaf_id_to_honest_av):
    X_test_leaf_ids = CART.apply(X_test)
    test_predictions = []
    for i in range(len(X_test_leaf_ids)):
        if(X_test_leaf_ids[i] in leaf_id_to_honest_av.keys()):
            test_predictions.append(leaf_id_to_honest_av[X_test_leaf_ids[i]])
        else:
            test_predictions.append(0.0)
    test_predictions = np.asarray(test_predictions)
    return test_predictions

def get_honest_test_MSE(CART,X_honest,y_honest,X_test,y_test):
    leaf_id_to_honest_av = get_honest_leaf_averages(CART,X_honest,y_honest)
    test_preds = get_honest_tree_test_preds(CART,X_test,y_test,leaf_id_to_honest_av)
    test_MSE = mean_squared_error(y_test,test_preds)
    return test_MSE

def get_honest_forest_test_MSE(RF,X_honest_y_honest,X_test,y_test):
   
    def mean(a):
        return sum(a) / len(a)
    
    n_tree = len(RF)
    all_tree_preds = []
    for i in range(n_tree):
        tree_leaf_id_to_honest_av = get_honest_leaf_averages(RF[i],X_honest,y_honest)
        tree_test_preds = get_honest_tree_test_preds(RF[i],X_test,y_test,leaf_id_to_honest_av)
        all_tree_preds.append(test_tree_preds)
    RF_honest_preds =  map(mean, zip(*all_tree_preds))
    return RF_honest_preds