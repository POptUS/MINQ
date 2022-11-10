import numpy as np


def pr01(name, x):
    """
    % function summe=pr01(name,x)
    % prints a (0,1) profile of x and returns the number of nonzeros
    """

    # check for row or column vector

    assert np.ndim(x) == 1, "x must be a vector"

    text = name + ": "
    summe = 0
    for k in range(len(x)):
        if x[k]:
            text += "1"
            summe = summe + 1
        else:
            text += "0"

    print(text + "   " + str(summe), " nonzeros")
    return summe
