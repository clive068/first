import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def L_i_vectorized (x, y, W):

    delta = 1.0
    scores = W.dot(x)

    margins = np.maximum (0, scores - scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return  loss_i

def L(X, Y, W):

    delta = 1.0
    scores = X.dot (W.T)
    margins = np.maximum (0, (scores.T - scores[np.arange(X.shape[0]), Y]).T + delta)
    margins[np.arange(X.shape[0]), Y] = 0
    return np.sum (margins)


if __name__ == "__main__":
    f = "/Users/cliveaustin/Downloads/cifar-10-batches-py/data_batch_1"
    tr = unpickle(f)
    Xtr_rows = tr["data"].reshape(tr["data"].shape[0], 32 * 32 * 3)
    Ytr_rows = np.asarray(tr["labels"])


    f = "/Users/cliveaustin/Downloads/cifar-10-batches-py/test_batch"
    test = unpickle(f)
    Xte_rows = test["data"].reshape(test["data"].shape[0], 32 * 32 * 3)
    Yte_rows = np.asarray(test["labels"])

    print("got the data")

    img = mpimg.imread('/Users/cliveaustin/Downloads/Downloads-icon.png')
    imgplot = plt.imshow(img)
    # plt.show()
    # W = np.zeros((10, Xtr_rows.shape[1]))
    W = np.random.rand (10, Xtr_rows.shape[1])

    print L(Xtr_rows, Ytr_rows, W)

    Xtr_image = Xtr_rows[4]
    Xtr_image = Xtr_image.reshape(1024, 3, order="F").reshape(32, 32, 3)
    Xtr_image = Xtr_rows[:49]
    Xtr_image = Xtr_image.reshape(49, 1024, 3, order="F").reshape(49, 32, 32, 3)
    imgplot = plt.imshow(Xtr_image[2])
    plt.show()
