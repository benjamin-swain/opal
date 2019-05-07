from keras.layers import Input, Dense
from keras.models import Model, model_from_json
from keras.optimizers import Adam
import numpy as np

# load pima indians dataset
text = np.loadtxt("nn_data.txt")

# Separate it into datapoints and labels
data = text[:, 0:26]
labels = text[:, 26:]

labels_float = text[:, 26:30]
labels_bool = text[:, 30:]

print(labels_float[0])
print(labels_bool[0])

print(data[0].shape)

num_inputs = len(data[0])
# num_outputs = len(labels[0])

num_output_floats = 4
num_output_bools = 3

# Aech stuffs
input_vector = Input(shape=(num_inputs,))

# 1 output layer
# input_layer = Dense(64, activation='tanh')(input_vector)
# hidden_layer = Dense(64, activation='tanh')(input_layer)
# output_layer = Dense(num_outputs, activation='linear')(hidden_layer)



# Aech example code
# inp = Input(shape=...)
# hidden1 = Dense(num_neurons0, activation='tanh')(inp)
# hidden2 = Dense(num_neurons1, activation='tanh')(hidden1)
# output1 = Dense(num_continuous_values, activation='linear')(hidden2)
# output2 = Dense(num_binary_values, activation='softmax')(hidden2)
#
# model = Model(inputs=(inp), outputs=(output1, output2))




# 2 output layers
input_layer = Dense(64, activation='tanh')(input_vector)
hidden_layer = Dense(64, activation='tanh')(input_layer)
output_layer = Dense(num_output_floats, activation='tanh')(hidden_layer)
output_layer2 = Dense(num_output_bools, activation='softmax')(hidden_layer)

model = Model(inputs=[input_vector], outputs=[output_layer, output_layer2])
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# model.fit(data, labels, validation_data, epochs=100, batch_size=32)
# model.fit(data, labels, epochs=5, batch_size=32)
model.fit(data, [labels_float, labels_bool], epochs=5, batch_size=32)

output = model.predict(np.array([[2047.840,2559.720,17.010,-0.297,-2.354,0.000,-12.581,-21.981,0.191,-0.001,-0.001,0.187,1.000,0.000,0.000,0.000,0.000,0.000,0.000,92.740,0.000,0.000,0.000,0.000,0.000,0.000]]))
# evaluate the model
# scores = model.evaluate(data, labels)

print('disp')
print(output)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# # later...
#
#
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
#
# # evaluate loaded model on test data
# loaded_model.compile(optimizer=Adam(), loss='categorical_crossentropy')
# score = loaded_model.evaluate(data, labels, verbose=0)
# print(score)
# # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



