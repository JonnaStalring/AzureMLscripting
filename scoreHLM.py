
# coding: utf-8

# In[ ]:


import pickle
import json
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model


def init():
    global model
    # note here "best_model" is the name of the model registered under the workspace
    # this call should return the path to the model.pkl file on the local disk.    
    fid = open("output\modelDump.pkl", "rb")
    model = pickle.load(fid)
    fid.close()


# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result = model.predict(data)

        # you can return any data type as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        result = str(e)
        return result

