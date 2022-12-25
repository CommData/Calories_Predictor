import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

X_test = pickle.load(open('data/X_test', 'rb'))
X_train = pickle.load(open('data/X_train', 'rb'))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = pickle.load(open('TrainedModel.sav', 'rb'))
# input_data = (0, 69, 179.0, 79.0, 5.0, 88.0, 38.7)
input_data = (1, 27, 154.0, 58.0, 10.0, 81.0, 39.8)
data_array = np.asarray(input_data)
data_reshaped = data_array.reshape(1, -1)
data_transformed = scaler.transform(data_reshaped)
print(data_transformed)
prediction_final = model.predict(data_transformed)
print(prediction_final)
