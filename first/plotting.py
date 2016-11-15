import numpy as np
import matplotlib.pyplot as plt


def plotLineStyle():
    X = np.linspace(0, 2 * np.pi, 50, endpoint=True)
    F1 = 3 * np.sin(X)
    F2 = np.sin(2 * X)
    F3 = 0.3 * np.sin(X)
    F4 = np.cos(X)
    plt.plot(X, F1, color="blue", linewidth=2.5, linestyle="-")
    plt.plot(X, F2, color="red", linewidth=1.5, linestyle="--")
    plt.plot(X, F3, color="green", linewidth=2, linestyle=":")
    plt.plot(X, F4, color="grey", linewidth=2, linestyle="-.")
    plt.show()


def borderLines():
    X = np.linspace(-2 * np.pi, 2 * np.pi, 70, endpoint=True)
    F1 = np.sin(2 * X)
    F2 = (2 * X ** 5 + 4 * X ** 4 - 4.8 * X ** 3 + 1.2 * X ** 2 + X + 1) * np.exp(-X ** 2)
    # get the current axes, creating them if necessary:
    ax = plt.gca()
    # making the top and right spine invisible:
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # moving bottom spine up to y=0 position:
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    # moving left spine to the right to position x == 0:
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.plot(X, F1)
    plt.plot(X, F2)
    plt.show()


def contourPlot():
    xlist = np.linspace(-3.0, 3.0, 3)
    ylist = np.linspace(-3.0, 3.0, 4)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X ** 2 + Y ** 2)
    plt.figure()
    cp = plt.contour(X, Y, Z)
    plt.clabel(cp, inline=True,
               fontsize=10)
    plt.title('Contour Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

def filledContourPlot():
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X**2 + Y**2)
    plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.title('Filled Contours Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

if __name__ == '__main__':
    plotLineStyle()
    borderLines()
    filledContourPlot()
    