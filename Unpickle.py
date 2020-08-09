# import libraries
import pandas as pd
import pickle
from ClassModel import Classifier # from ClassModel.py

# unpickle classifier    
class customUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if name=="Classifier":
            return Classifier
        return super().find_class(module,name)
    
with open('modelpicklefinal', 'rb') as modelnb:
    model = customUnpickler(modelnb).load()

# new csv file for predictions    
prediction_file = 'Predict.csv'
pred = pd.read_csv(prediction_file)
T = pred['TEXTS']

# classify text
model.classify(T)






