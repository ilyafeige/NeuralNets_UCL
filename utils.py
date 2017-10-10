'''Private utility functions for neural networks notebooks.'''

#
# Only use the notebooks.
# This file should not be opened.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


from io import BytesIO
import itertools
import gzip
import matplotlib.pyplot as plt
import numpy as np
import pickle
import requests
import time


def read_mnist(dataset='train', flatten=True):
    """Return an iterator of images and labels.

    Returns an iterator of 2-tuples with the first element a
    numpy.float32 array of pixel data for the given image and
    the second element the correspnding label as numpy.int64.

    If flatten, each image array has shape (784,), otherwise
    each image array has shape (28, 28).
    """
    MNIST_PATH = ('http://www.iro.umontreal.ca/~lisa'
                  '/deep/data/mnist/mnist.pkl.gz')
    response = requests.get(MNIST_PATH)
    content = BytesIO(response.content)
    with gzip.open(content, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='bytes')

    if 'train' in dataset.lower():
        images, labels = train_set
    elif 'valid' in dataset.lower():
        images, labels = valid_set
    elif 'test' in dataset.lower():
        images, labels = test_set
    else:
        raise ValueError("dataset must be 'train', 'valid' or 'test'. "
                         "Got '{}'".format(dataset))
    if not flatten:
        images = images.reshape(-1, 28, 28)

    return images, labels


def show(image, label=None):
    """Render a given numpy.float32 array of pixel data."""
    fig, ax = plt.subplots()
    plot = ax.imshow(image.reshape(28, 28),
                     cmap=plt.cm.gray)
    plot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    # ensure label 0 is not passed as False
    if label or label == 0.:
        ax.set_xlabel("Label: {}".format(label),
                      size=14)
    plt.show()


def one_hot_encode(label):
    """One-hot encode a class label.

    Given a class label, return an array of length 10 with all
    elements zero, except for the element with index `label`,
    which is 1.0.
    """
    encoded = np.zeros(10)
    encoded[label] = 1.0
    return encoded


def softmax(z):
    """Return the softmax of vector z.

    The softmax returns normalized positive values.
    It is defined as softmax(z_i) = normalise(exp(z_i))
    = \\frac{exp(z_i)}{\sum_j exp(z_j)}.
    """
    z -= np.max(z)  # for numerical stability
    exps = np.exp(z)
    return np.divide(exps, np.sum(exps))


def dloss_dw(y_true, x, y_pred):
    """Analytic gradient of cross-entropy with reference to matrix W."""
    return np.outer(np.negative(x), y_true*(1-y_pred))


def dloss_db(y_true, x, y_pred):
    """Analytic gradient of cross-entropy with reference to vector b"""
    return np.negative(y_true)*(1-y_pred)


def cross_entropy_loss(p, q):
    """Return the cross-entropy of q with respect to p.

    The cross-entropy is a measure of how much distribution q
    diverges from distribution p. It is defined as
    H_p(q) = - \sum_x p(x) \log q(x).
    """
    return -np.sum(p * np.log(q))


def neural_network_prediction(x, W, b, clip=True):
    """This is the entire neural net!"""
    y_scores = (np.dot(x, W) + b).ravel()  # flatten
    prediction = softmax(y_scores)

    if clip:  # avoid 0s and 1s
        eps = 1e-10
        return prediction.clip(eps, 1 - eps)
    else:
        return prediction


def learn_mnist(W, b, max_iters=100000, learning_rate=0.01, batch_size=20):
    """Update W, b to reduce cross-entropy on using gradient descent."""
    W = W.copy()  # don't overwrite original parameters
    b = b.copy()
    grad_w = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)

    # keep track of some metrics
    total_loss = 0
    losses = []
    train_errors = []
    test_errors = []

    start = time.time()
    print_freq = max_iters / 10

    # learn by cycling through data for max_iters iterations
    X, Y = read_mnist('train')
    
    for i, (x, label) in enumerate(itertools.cycle(zip(X, Y))):
        y_true = one_hot_encode(label)
        y_pred = neural_network_prediction(x, W, b)
        total_loss += cross_entropy_loss(y_true, y_pred)

        # compute gradient of loss w.r.t. params
        grad_w += dloss_dw(y_true, x, y_pred)
        grad_b += dloss_db(y_true, x, y_pred)

        # update parameters every batch_size iterations
        if i % batch_size == 0:
            grad_w /= batch_size
            grad_b /= batch_size
            W -= learning_rate * grad_w  # <- updates
            b -= learning_rate * grad_b  # <- updates
            # reset gradients for next batch
            grad_w = np.zeros_like(W)
            grad_b = np.zeros_like(b)

        # print some metrics
        if i % print_freq == 0:
            total_loss /= print_freq
            train_error = error_rate_mnist(W, b, 'train')
            test_error = error_rate_mnist(W, b, 'test')
            running_time = (time.time() - start)
            print(("Iteration {i} | Loss: {loss:.4f} | "
                   "Train error: {train:.4f} | Test error: {test:.4f} | "
                   "Total run time: {t:.1f}s".format(i=i,
                                                     loss=total_loss,
                                                     train=train_error,
                                                     test=test_error,
                                                     t=running_time)))
            # accumulate the metrics
            train_errors.append(train_error)
            test_errors.append(test_error)
            losses.append(total_loss)
            total_loss = 0

        if i > max_iters:
            return W, b, losses, train_errors, test_errors


def plot_learning(losses, train_errors, test_errors):
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5), squeeze=True)

    ax[0].plot(range(len(train_errors)), train_errors, '-o', label='Training error')
    ax[0].plot(range(len(test_errors)),  test_errors, '-o', label='Test error')
    ax[0].set_ylim(0, 1)
    ax[0].legend(loc='upper right')
    ax[0].set_title('Error rate')

    ax[1].plot(range(len(losses)), losses, '-o')
    ax[1].set_title('Losses')
    plt.show()


def error_rate_mnist(W, b, dataset='test'):
    """Calculate the error rate."""
    data_seen = 0.
    total_correct = 0.
    total_errors = 0.

    X, Y = read_mnist(dataset)
    for x, y in zip(X, Y):
        y_pred = np.argmax(neural_network_prediction(x, W, b))
        if y_pred == y:
            total_correct += 1
        else:
            total_errors += 1
        data_seen += 1

    return total_errors / data_seen
