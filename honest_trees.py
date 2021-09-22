import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


'''
This python script is used to compute predictions and MSE for honest trees and forests. For empty cells we use average over parent node instead. 

'''

def flatten_list(t):
    return [item for sublist in t for item in sublist]


def get_test_prediction(decision_path,node_id_to_honest_av,node_id_to_honest_count):
    test_pred = 0.0
    for node_id in decision_path[::-1]:
        if node_id_to_honest_count[node_id] == 0:
            continue
        else:
            test_pred = node_id_to_honest_av[node_id]
            break
    return test_pred
    

def get_all_decision_paths(CART,X_honest):
    
    '''
    
    This method returns 1. the decision path of each sample and 2. all node_ids used in decision paths for X_honest
    
    '''
    node_indicator = CART.decision_path(X_honest)
    leaf_id = CART.apply(X_honest)
    sample_id_to_decision_path = {}
    node_ids = []
    for i in range(len(X_honest)):
        sample_id = i 
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]
        sample_id_to_decision_path[i] = node_index
        node_ids.append(node_index)
    return sample_id_to_decision_path,np.unique(np.array(flatten_list(node_ids)))
    
    
def get_honest_leaf_averages(CART,X_honest,y_honest):
    X_honest_decsion_paths,X_honest_node_ids = get_all_decision_paths(CART,X_honest)    
    node_id_to_honest_av = {}
    node_id_to_honest_count = {}
    all_node_ids = range(CART.tree_.node_count)
    for node_id in all_node_ids:
        if node_id in X_honest_node_ids:
            honest_sample_ids_at_node = [sample_id for sample_id,decision_path in X_honest_decsion_paths.items() if node_id in decision_path]
            node_id_to_honest_av[node_id] = y_honest[honest_sample_ids_at_node].mean()
            node_id_to_honest_count[node_id] = len(honest_sample_ids_at_node)
        else:
            node_id_to_honest_av[node_id] = 'nan'
            node_id_to_honest_count[node_id] = 0

    return node_id_to_honest_av,node_id_to_honest_count





def get_honest_tree_test_preds(CART,X_test,y_test,node_id_to_honest_av,node_id_to_honest_count):
    X_test_decision_paths = get_all_decision_paths(CART,X_test)[0]
    test_preds = []
    #count = 0
    for i in range(len(X_test_decision_paths)):
        test_sample_decision_path = X_test_decision_paths[i]
        test_sample_pred = get_test_prediction(test_sample_decision_path,node_id_to_honest_av,node_id_to_honest_count)
        test_preds.append(test_sample_pred)
    return test_preds

def get_honest_test_MSE(CART,X_honest,y_honest,X_test,y_test):
    node_id_to_honest_av,node_id_to_honest_count = get_honest_leaf_averages(CART,X_honest,y_honest)
    test_preds = get_honest_tree_test_preds(CART,X_test,y_test,node_id_to_honest_av,node_id_to_honest_count)
    test_MSE = mean_squared_error(test_preds,y_test)
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