import numpy as np
import pickle
from xgboost import XGBClassifier
# loading the saved model
loaded_model = pickle.load(open('C:/Users/user/OneDrive/Desktop/Employee_churn/train_model.sav', 'rb'))
input_data = (0.38,0.53,2,157,3,0,0,True,False)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person stayed in the company')
else:
  print('The person left the company')