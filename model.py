from tensorflow import keras

def model_and_weight():

    #load JSON and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)

    #load pretrained weights into the model
    loaded_model.load_weights('model_weights.h5')

    print("Loaded Model Architecture and Pretrained Weights")

    return loaded_model



