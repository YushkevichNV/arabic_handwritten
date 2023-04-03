import numpy as np
import pandas as pd
from tensorflow import keras


arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                    'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                    'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

x_test = pd.read_csv("Arabic Handwritten Characters Dataset CSV/csvTestImages 3360x1024.csv",
                     header=None).to_numpy()    #your images
x_test = x_test.reshape(-1,32,32,1)
x_test = x_test / 255.0

model = keras.models.load_model('model_ewaran.h5')

y_preds = model.predict(x_test)
y_pred_classes = np.argmax(y_preds, axis=1)
print(list(map(lambda x: arabic_characters[x] , y_pred_classes)))