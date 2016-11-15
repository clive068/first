import numpy as np
import matplotlib.pyplot as plt

def generateData():
    X = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8')

    for j in xrange(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.5
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return X, y

def plotClassifier():
    # plot the resulting classifier
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

if __name__ == '__main__':
    N = 100
    D = 2
    K = 5
    X, y = generateData()
    num_examples = X.shape[0]

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

    # initialize parameters randomly
    h = 40  # size of hidden layer
    W = 0.01 * np.random.randn(D, h)
    b = np.zeros((1, h))
    W2 = 0.01 * np.random.randn(h, K)
    b2 = np.zeros((1, K))

    for i in xrange(5000):

        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1)[:, None]
        # have to use [:, None ] instead of np.sum(exp_scores, axis=1, keepdims = True) otherwise broadcasting doesn't work
        # (300, ) ratherthan (300,1)

        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5 * reg * np.sum (W*W)
        loss = data_loss + reg_loss
        if i % 100 == 0:
            print "iteration %d: loss %f" % (i, loss)

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0)[None, :]
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0)[None, :]

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

    # evaluate training set accuracy
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print 'training accuracy: %.4f' % (np.mean(predicted_class == y))
    plotClassifier()
    plt.show()
    print ("Finished")
