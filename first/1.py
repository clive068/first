import numpy as np
import matplotlib.pyplot as plt


def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable
        print __name__

    # Instance method
    def greet(self, loud=False):
        print __name__
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name


def numpyExpanple():
    a = np.array([1, 2, 3])  # Create a rank 1 array
    print type(a)  # Prints "<type 'numpy.ndarray'>"
    print a.shape  # Prints "(3,)"
    print a[0], a[1], a[2]  # Prints "1 2 3"
    a[0] = 5  # Change an element of the array
    print a  # Prints "[5, 2, 3]"

    b = np.array([[1, 2, 3], [4, 5, 6]])  # Create a rank 2 array
    print b.shape  # Prints "(2, 3)"
    print b[0, 0], b[0, 1], b[1, 0]


def numpyRandomExample():
    a = np.zeros((2, 2))  # Create an array of all zeros
    print a  # Prints "[[ 0.  0.]
    #          [ 0.  0.]]"

    b = np.ones((1, 2))  # Create an array of all ones
    print b  # Prints "[[ 1.  1.]]"

    c = np.ones((2, 2))  # Create a constant array
    print c * 7 # Prints "[[ 7.  7.]
    #          [ 7.  7.]]"

    d = np.eye(2)  # Create a 2x2 identity matrix
    print d  # Prints "[[ 1.  0.]
    #          [ 0.  1.]]"

    e = np.random.random((2, 2))  # Create an array filled with random values
    print e  # Might print "[[ 0.91940167  0.08143941]


def numpyArrayIndexing():
    # Create the following rank 2 array with shape (3, 4)
    # [[ 1  2  3  4]
    #  [ 5  6  7  8]
    #  [ 9 10 11 12]]
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Use slicing to pull out the subarray consisting of the first 2 rows
    # and columns 1 and 2; b is the following array of shape (2, 2):
    # [[2 3]
    #  [6 7]]
    b = a[2:, :3]

    # A slice of an array is a view into the same data, so modifying it
    # will modify the original array.
    print a[0, 1]  # Prints "2"
    b[0, 0] = 77  # b[0, 0] is the same piece of data as a[0, 1]
    print a[0, 1]  # Prints "77"
    # Create the following rank 2 array with shape (3, 4)
    # [[ 1  2  3  4]
    #  [ 5  6  7  8]
    #  [ 9 10 11 12]]

    # Two ways of accessing the data in the middle row of the array.
    # Mixing integer indexing with slices yields an array of lower rank,
    # while using only slices yields an array of the same rank as the
    # original array:
    row_r1 = a[1, :]    # Rank 1 view of the second row of a
    row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
    print row_r1, row_r1.shape  # Prints "[5 6 7 8] (4,)"
    print row_r2, row_r2.shape  # Prints "[[5 6 7 8]] (1, 4)"

    # We can make the same distinction when accessing columns of an array:
    col_r1 = a[:, 1]
    col_r2 = a[:, 1:2]
    print col_r1, col_r1.shape  # Prints "[ 2  6 10] (3,)"
    print col_r2, col_r2.shape  # Prints "[[ 2]
                                #          [ 6]
                                #          [10]] (3, 1)"

def numpyPlotting():
    # Compute the x and y coordinates for points on sine and cosine curves
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Plot the points using matplotlib
    plt.plot(x, y_sin)
    plt.plot(x, y_cos)
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    plt.title('Sine and Cosine')
    plt.legend(['Sine', 'Cosine'])
    plt.show()


def numpyIntegerIndexing():
    # Create the following rank 2 array with shape (3, 4)
    # [[ 1  2  3  4]
    #  [ 5  6  7  8]
    #  [ 9 10 11 12]]
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Two ways of accessing the data in the middle row of the array.
    # Mixing integer indexing with slices yields an array of lower rank,
    # while using only slices yields an array
    #
    #of the same rank as the
    # original array:
    row_r1 = a[1, :]  # Rank 1 view of the second row of a  
    row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
    print row_r1, row_r1.shape  # Prints "[5 6 7 8] (4,)"
    print row_r2, row_r2.shape  # Prints "[[5 6 7 8]] (1, 4)"

    # We can make the same distinction when accessing columns of an array:
    col_r1 = a[:, 1]
    col_r2 = a[:, 1:2]
    print col_r1, col_r1.shape  # Prints "[ 2  6 10] (3,)"
    print col_r2, col_r2.shape  # Prints "[[ 2]
    #          [ 6]
    #          [10]] (3, 1)"


def numpyMutate():
    print __name__
    # Create a new array from which we will select elements
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


    # Create an array of indices
    b = np.array([0, 2, 0, 1])

    # Select one element from each row of a using the indices in b
    print a[np.arange(4), b]  # Prints "[ 1  6  7 11]"

    # Mutate one element from each row of a using the indices in b
    a[np.arange(4), b] += 10

    print a  # prints "array([[11,  2,  3],
    #                [ 4,  5, 16],
    #                [17,  8,  9],
    #                [10, 21, 12]])


def numpyBoolean():
    a = np.array([[1, 2], [3, 4], [5, 6]])

    bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
    # this returns a numpy array of Booleans of the same
    # shape as a, where each slot of bool_idx tells
    # whether that element of a is > 2.

    print bool_idx  # Prints "[[False False]
    #          [ True  True]
    #          [ True  True]]"

    # We use boolean array indexing to construct a rank 1 array
    # consisting of the elements of a corresponding to the True values
    # of bool_idx
    print a[bool_idx]  # Prints "[3 4 5 6]"

    # We can do all of the above in a single concise statement:
    print a[a > 2]  # Prints "[3 4 5 6]"


def numpyDataType():

    x = np.array([1, 2])  # Let numpy choose the datatype
    print x.dtype         # Prints "int64"

    x = np.array([1.0, 2.0])  # Let numpy choose the datatype
    print x.dtype             # Prints "float64"

    x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
    print x.dtype                         # Prints "int64"s


    x = np.array([1.1, 2.0], dtype=np.int64)  # Force a particular datatype
    print x.dtype  # Prints "int64"s


def numpyDotProduct():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])

    v = np.array([9, 10])
    w = np.array([11, 12])

    # Inner product of vectors; both produce 219
    print v.dot(w)
    print np.dot(v, w)

    # Matrix / vector product; both produce the rank 1 array [29 67]
    print x.dot(v)
    print np.dot(x, v)

    # Matrix / matrix product; both produce the rank 2 array
    # [[19 22]
    #  [43 50]]
    print x.dot(y)
    print np.dot(x, y)
    print np.sum(x)  # Compute sum of all elements; prints "10"
    print np.sum(x, axis=0)  # Compute sum of each column; prints "[4 6]"
    print np.sum(x, axis=1)  # Compute sum of each row; prints "[3 7]"


    v1= np.sum(x)  # Compute sum of all elements; prints "10"
    v2= np.sum(x, axis=0)  # Compute sum of each column; prints "[4 6]"
    v3= np.sum(x, axis=1)  # Compute sum of each row; prints "[3 7]"


def numpyBroadcasting():
    # Compute outer product of vectors
    v = np.array([1, 2, 3])  # v has shape (3,)
    w = np.array([4, 5])  # w has shape (2,)
    # To compute an outer product, we first reshape v to be a column
    # vector of shape (3, 1); we can then broadcast it against w to yield
    # an output of shape (3, 2), which is the outer product of v and w:
    # [[ 4  5]
    #  [ 8 10]
    #  [12 15]]
    print np.reshape(v, (3, 1)) * w

    # Add a vector to each row of a matrix
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
    # giving the following matrix:
    # [[2 4 6]
    #  [5 7 9]]
    print x + v

    # Add a vector to each column of a matrix
    # x has shape (2, 3) and w has shape (2,).
    # If we transpose x then it has shape (3, 2) and can be broadcast
    # against w to yield a result of shape (3, 2); transposing this result
    # yields the final result of shape (2, 3) which is the matrix x with
    # the vector w added to each column. Gives the following matrix:
    # [[ 5  6  7]
    #  [ 9 10 11]]
    print (x.T + w).T
    # Another solution is to reshape w to be a row vector of shape (2, 1);
    # we can then broadcast it directly against x to produce the same
    # output.
    print x + np.reshape(w, (2, 1))

    # Multiply a matrix by a constant:
    # x has shape (2, 3). Numpy treats scalars as arrays of shape ();
    # these can be broadcast together to shape (2, 3), producing the
    # following array:
    # [[ 2  4  6]
    #  [ 8 10 12]]
    print x * 2


if __name__ == '__main__':
    q = 30
    print q
    print sign(4)
    # clive=Greeter("clive")
    # clive.greet()
    # numpyExpanple()
    # numpyRandomExample()
    # numpyArrayIndexing()
    # numpyPlotting()
    # numpyIntegerIndexing()
    numpyMutate()
    numpyBoolean()
    numpyDataType()
    numpyDotProduct()
    numpyBroadcasting()
    

    