import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--problem_type', type=str, default='or', help='"or" | "and" | "xor" | "nand"')
parser.add_argument('-n', '--neurons', type=int, default=1)
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('-v', '--verbosity', type=int, default=0, help='0 | 1')
args = parser.parse_args()

print(f'Problem type: "{args.problem_type}"')
print(f'Neuron(s) in hidden layer: {args.neurons}')
print(f'Epochs: {args.epochs}')

training_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

def get_training_labels(problem_type):
    if problem_type == 'or':
        return np.array([0, 1, 1, 1])
    elif problem_type == 'and':
        return np.array([0, 0, 0, 1])
    elif problem_type == 'xor':
        return np.array([0, 1, 1, 0])
    elif problem_type == 'nand':
        return np.array([1, 1, 1, 0])
    else:
        return np.array([0, 0, 0, 0])

training_labels = get_training_labels(args.problem_type)

model = keras.Sequential([
    keras.layers.Dense(units=args.neurons, activation='relu', input_dim=2),
    keras.layers.Dense(units=2, activation='softmax')
])

optimizer = tf.train.AdamOptimizer()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

history = model.fit(
    training_data,
    training_labels,
    epochs=args.epochs,
    verbose=args.verbosity)

testing_data = training_data
testing_labels = training_labels

_, acc = model.evaluate(testing_data, testing_labels, verbose=0)
print(f'Testing data accuracy = {acc * 100}%')

predictions = model.predict(testing_data)

print()
print('Input Expected Actual')
print('----- -------- ------')

for i, _ in enumerate(range(len(testing_data))):
    print(f'{testing_data[i]} {str(testing_labels[i]).rjust(8)} {str(np.argmax(predictions[i])).rjust(6)}')
