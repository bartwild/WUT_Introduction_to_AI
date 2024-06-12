from data_loader import load_data, load_data2
from network import Network
import random
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist

# from validation_tests import best_batch_size, best_epochs, best_lambda, best_learnrate, plot

image_size = 28*28  # 8x8
random.seed(2137)
"""
training_data, test_data = load_data2(image_size)
net = Network([image_size, 50, 25, 10, 10])
time_start = time.time()
net.stochastic_gradient_descent(
    training_data=training_data,
    test_data=test_data,
    epochs=20,
    batch_size=15,
    learn_rate=1.0,
    lambda_=1.0,
    show_confusion_matrix=True,
    fun="sigmoid"
)
print(time.time() - time_start)
"""
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_label_counts = [list(train_labels).count(i) for i in range(10)]
test_label_counts = [list(test_labels).count(i) for i in range(10)]
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].bar(range(10), train_label_counts, color='blue')
axs[0].set_title('Rozkład cyfr w zbiorze treningowym MNIST')
axs[0].set_ylabel('Liczba wystąpień')
axs[1].bar(range(10), test_label_counts, color='green')
axs[1].set_title('Rozkład cyfr w zbiorze testowym MNIST')
axs[1].set_ylabel('Liczba wystąpień')
axs[1].set_xlabel('Cyfry')
plt.tight_layout()
plt.show()
"""
"""
# Standard data / independent values
epochs = 40
batch_size = 15
learn_rate = 0.1
lambd = 5.0

# Testing data / dependent values
testing_epochs = [
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 50, 55, 60, 65, 70, 75, 80,
    85, 90, 95, 100]
testing_batchsize = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                     20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50]
testing_learnrate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
testing_lambda = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                  8.5, 9.0, 9.5, 10.0, 12.0, 15.0, 20.0, 40.0]

# Best epochs:
best_epoch, acs = best_epochs(net, training_data, validation_data, testing_epochs, batch_size, learn_rate, lambd)
plot("Epochs", testing_epochs, acs)

# Best batch size:
best_batch, acs = best_batch_size(net, training_data, validation_data, epochs, testing_batchsize, learn_rate, lambd)
plot("Batch size", testing_batchsize, acs)

# Best learn rate:
best_lr, acs = best_learnrate(net, training_data, validation_data, epochs, batch_size, testing_learnrate, lambd)
plot("Learn rate", testing_learnrate, acs)

# Best lambda:
best_lbd, acs = best_lambda(net, training_data, validation_data, epochs, batch_size, learn_rate, testing_lambda)
plot("Lambda", testing_lambda, acs)

print(f"Epochs: {best_epoch}\nBatch size: {best_batch}\nLearn rate: {best_lr}\nLambda: {best_lbd}\n")
net.stochastic_gradient_descent(
    training_data=training_data,
    test_data=test_data,
    epochs=best_epoch,
    batch_size=best_batch,
    learn_rate=best_lr,
    lambda_=best_lbd
)
"""
