import numpy as np
import matplotlib.pyplot as plt

def preProcessing ():
    X = 2 * np.random.rand(300, 2)
    plt.scatter(X[:,0], X[:, 1], color="blue")
    Y = X - np.mean(X, axis=0)
    plt.scatter(Y[:, 0], Y[:, 1], color="red")
    Z = Y - Y / np.std(Y, axis=0)
    plt.scatter(Z[:, 0], Z[:, 1], color="green")
    plt.show()

def covariance():
    X = 5 * np.random.rand(300, 2)
    plt.scatter(X[:,0], X[:, 1], color="blue")
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    cov= X.T.dot(X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    #print "U= ", U
    #print "S= ", S
    #print "V= ", V

    Xrot = X.dot(U)
    plt.scatter(Xrot[:, 0], Xrot[:, 1], color="green")
    Xrot1D = X.dot(U[:,:1])
    plt.scatter(Xrot1D[:, 0], Xrot1D[:, 0],  color="red")
    plt.show()

    return cov

if __name__ == "__main__":
    W = np.random.randn (5, 10)
    X = np.random.randn (10, 3)
    D = W.dot(X)

    dD = np.random.randn (*D.shape)
    dW = dD.dot(X.T)
    dX = W.T.dot(dD)



    # preProcessing()

    c = covariance()



