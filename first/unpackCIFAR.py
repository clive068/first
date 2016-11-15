import numpy as np

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train (self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros (num_test, dtype = self.ytr.dtype)

        for i in xrange(num_test):
            if (i % 100 == 0):
                print ('.')
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred


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

    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr_rows)
    Yte_predict = nn.predict(Xte_rows)

    print ('accuracy %f ' (np.mean(Yte_predict == Yte_rows)))