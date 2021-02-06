from BaseModel import BaseModel
import numpy as np
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report


class NNModel(BaseModel):
    def __init__(self,*args, **kwargs):
        super(NNModel,self).__init__(*args, **kwargs)
        self.checkpoint = ModelCheckpoint(f'{self.save_folder}/{self.model_name}.h5', monitor='val_loss', verbose=1,save_best_only=True, mode='auto')

    def build_model(self,input_shape):
        optimizer = keras.optimizers.Adam(lr=0.001)
        
        METRICS = [
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision')
        ]
        
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(32, input_shape=input_shape)) # X_train.shape[1:]
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.2))
        
        
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(32))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(32))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.2))
        
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=METRICS)
        model.summary()
        self.clf = model
        return self.clf

    def train(self,X_train,y_train,X_test,y_test):
        X_train_encoded, y_train_encoded = super().train(X_train,y_train)
        self.build_model(X_train_encoded.shape[1:])
        print("Training")
        # Encoding the Target
        y_test_encoded = y_test
        # Transform the features
        X_test_encoded = self.column_transformer.transform(X_test)
        self.clf.fit(X_train_encoded,y_train_encoded,validation_data=(X_test_encoded,y_test_encoded),epochs=50,callbacks=[self.checkpoint],verbose=1,batch_size=5)
        print("Training Finished")


    def load_model(self):
        self.clf = keras.models.load_model(f"{self.save_folder}/{self.model_name}.h5")
        super().load_model()
