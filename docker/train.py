# MNIST trainign script
# The MNIST (Modified National Institute of Standards and Technology)
# dataset is a collection of 70,000 grayscale images of handwritten 
# digits (0 through 9), each sized at 28x28 pixels.
# It's widely used as a benchmark for image classification tasks
# in machine learning.

# The training script utilizes TensorFlow and its Keras API to build,
# train, and evaluate a simple neural network model on the MNIST dataset.

import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def main():
    parser = argparse.ArgumentParser()
    # specifies the learning rate, controlling how much the model 
    # adjusts during training.
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    # determines the number of samples processed before the model's 
    # internal parameters are updated.
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # next code is used to save the model after the katib hyperparameters tuning
    # parser.add_argument('--model_path', type=str, default='trained_model.h5', help='Path to save the trained model')
    args = parser.parse_args()

    # Load and preprocess MNIST data
    # Loads the dataset, splitting it into training and testing sets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reshaping: Flattens each 28x28 image into a 784-element vector
    # to feed into the neural network.
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    # Normalization: Scales pixel values from the range [0, 255] to [0, 1]
    # to improve training efficiency and stability.
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

    # Define a simple neural network model
    # models.Sequential: Creates a linear stack of layers for the model.
    model = models.Sequential([
        # First Dense Layer: Contains 128 neurons with ReLU activation,
        # introducing non-linearity and enabling the model to learn complex
        # patterns.
        layers.Dense(128, activation='relu', input_shape=(784,)),
        # Second Dense Layer: Contains 10 neurons with softmax activation,
        # producing a probability distribution over the 10 digit classes.
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    # optimizer: Uses the Adam optimization algorithm with the specified
    #            learning rate for efficient training.
    # loss: Employs sparse categorical crossentropy, suitable for multi-class
    #       classification problems with integer labels.
    # metrics: Tracks accuracy during training and evaluation to monitor
    #          performance.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    # model.fit(): Trains the model on the training data for a specified number of epochs and batch size.
    # epochs=3: Runs the training process three times over the entire dataset.
    # verbose=0: Suppresses output during training for cleaner logs.
    model.fit(x_train, y_train, epochs=3, batch_size=args.batch_size, verbose=0)

    # Evaluate the model
    # model.evaluate(): Assesses the model's performance on the test dataset,
    #                   returning the loss and accuracy metrics.
    # verbose=0: Suppresses output during evaluation.
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Output the accuracy in the format expected by Katib
    print(f'accuracy={accuracy}')

    # Once we find the hyperparameters tuning output,
    # we can save the optimized model
    # model.save(args.model_path)

if __name__ == '__main__':
    main()
