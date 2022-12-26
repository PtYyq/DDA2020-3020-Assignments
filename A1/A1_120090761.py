import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_120090761(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :w type: numpy.ndarray
    :XT type: numpy.ndarray
    :InvXTX type: numpy.ndarray
   
    """
    # your code goes here
    XT = np.transpose(X)
    XTX = np.dot(XT,X)
    try:
        InvXTX = np.linalg.inv(XTX)
        w = np.dot(np.dot(InvXTX,XT),y)
        # return in this order
    except:
        InvXTX = None
        w = None
    return w, XT, InvXTX