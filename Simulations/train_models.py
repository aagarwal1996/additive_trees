import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from generate_data import *

sys.path.append("../")
from honest_trees import *
'''

This script is used to train the following models: 

1.  honest and non-honest CART with min_sample_leaf condition (default = 5)
2.  honest and non-honest CART with CCP with CV 
'''


def CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = False):
    if honest == False:
        CART = DecisionTreeRegressor(min_samples_leaf = 5)
        CART.fit(X_train,y_train)
        CART_preds = CART.predict(X_test)
        return mean_squared_error(CART_preds,y_test)
    else:
        CART = DecisionTreeRegressor(min_samples_leaf = 5)
        CART.fit(X_train,y_train)
        honest_test_mse = get_honest_test_MSE(CART,X_honest,y_honest,X_test,y_test)
        return honest_test_mse
        

def CART_CCP(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,k = 5):
    id_threshold = sigma**2/len(X_train)
    alphas = np.geomspace(0.1*id_threshold, 1000*id_threshold, num=5)
    scores = []
    models = []
    for alpha in alphas:
        CART = DecisionTreeRegressor(min_samples_leaf = 5,ccp_alpha = alpha)
        CART.fit(X_train,y_train)
        models.append(CART)
        scores.append(cross_val_score(CART, X_train, y_train, cv=k).mean())
        best_CART = models[scores.index(max(scores))]
        dishonest_MSE = mean_squared_error(best_CART.predict(X_test),y_test)
        honest_MSE = get_honest_test_MSE(best_CART,X_honest,y_honest,X_test,y_test)
        return honest_MSE,dishonest_MSE
        
def train_all_models(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,k = 5):
    honest_CART =  CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = True)
    dishonest_CART = CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = False)
    #honest_CART_CCP,dishonest_CART_CCP = CART_CCP(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,k = 5)
    return honest_CART,dishonest_CART
#,honest_CART_CCP
#,,dishonest_CART_CCP
    

        
        