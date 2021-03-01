from BaseModel import BaseModel
from xgboost import XGBClassifier,plot_importance
import numpy as np

class XGBModel(BaseModel):
    def __init__(self,*args, **kwargs):
        super(XGBModel,self).__init__(*args, **kwargs)
        self.clf = XGBClassifier()

    
    def train(self,X_train,y_train):
        X_train_encoded, y_train_encoded = super().train(X_train,y_train)
        print("Training")
        self.clf.fit(X_train_encoded,y_train_encoded)
        print("Training Finished")

