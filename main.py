import pickle

from cryptography.fernet import Fernet
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(7, 4), random_state=5, solver="sgd",
                                        learning_rate="constant", learning_rate_init=0.00001)

# encrypting the model
model_bytes = pickle.dumps(classifier)
key = Fernet.generate_key()
f = Fernet(key)
encrypted_model = f.encrypt(model_bytes)
print("done")

# decrypting the model
decrypted_model = f.decrypt(encrypted_model)
model = pickle.loads(decrypted_model)
print("done")

