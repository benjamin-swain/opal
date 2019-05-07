#from keras.models import model_from_json
#from keras.models import model_from_yaml
# from keras.optimizers import Adam
# from keras.layers import Input, Dense
# from keras.models import Model, model_from_json
# from keras.optimizers import Adam
# import numpy as np

from keras.models import load_model

# load keras model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer=Adam(), loss='categorical_crossentropy')
#print("Loaded model from disk")