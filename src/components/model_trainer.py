import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=True),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            params={
                "Decision Tree":{
                    "criterion":['squared_error','friedman_mse','absolute_error','poisson']
                },
                
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features':['sqrt','log2',None],
                },
                
                "Gradient Boosting":{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9]
                },

                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01,0.05,0.1],
                    'iterations': [30,50,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate': [.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            # To get the best model score and name here from the model_report dict.
            model_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            # Get the best parameters for the selected model
            best_params = params[best_model_name]
            
            if len(best_params) > 0:
                # If there are parameters to tune, use GridSearchCV to find the best ones
                gs = GridSearchCV(best_model, best_params, cv=3)
                gs.fit(X_train, y_train)
                
                # Update the model with best parameters and retrain
                best_model.set_params(**gs.best_params_)
            
            # Final training of the best model
            best_model.fit(X_train, y_train)

            if best_model_score<0.6:
                raise CustomException("No best Model found")
            
            logging.info(f"Best Found model: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)

