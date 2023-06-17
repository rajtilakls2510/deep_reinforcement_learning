import os, json
from tensorflow.keras import Model, optimizers
import tensorflow as tf

# @tf.function
def set_weights(trainable_weights, weights):
    for var, weight in zip(trainable_weights, weights):
        var.assign(weight)


# @tf.function
def get_weights(trainable_weights):
    return [tf.convert_to_tensor(u) for u in trainable_weights]


def save_keras_model(model, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        print(model.get_config())
        f.write(json.dumps(model.get_config()))

    model.save_weights(os.path.join(path, "weights"))
    with open(os.path.join(path, "optimizer"), "w") as f:
        print(optimizers.serialize(model.optimizer))
        f.write(json.dumps(optimizers.serialize(model.optimizer)))


def load_keras_model(path):
    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.loads(f.read())
    model = Model().from_config(config)
    model.load_weights(os.path.join(path, "weights"))
    with open(os.path.join(path, "optimizer"), "r") as f:
        model.compile(optimizer=optimizers.deserialize(json.loads(f.read())))
    return model