import numpy as np
import math

def ICC3_1(label,Y):
    """
    code for ICC computation applied in the manuscript
    based on https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py.
    :param label: A label or identifier (probably a string) for the given Y. 
        This is used to return the result in a dictionary format.
    :param Y: A 2D numpy array where rows represent subjects and columns represent sessions.
        Each entry in the matrix represents a measurement (time-series feature in our study) 
        for a given subject in a specific session.
    :return: A dictionary with the label as the key and the ICC as the value.
    """
    
    [nb_subjects, nb_sessions] = Y.shape
    
    # Degrees of freedom
    dfc = nb_sessions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(nb_sessions), np.ones((nb_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(nb_subjects), (nb_sessions, 1))  # subjects
    X = np.hstack([x, x0])
    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F"))
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()
    MSE = SSE / dfe
    # Sum square session effect - between columns（sessions）
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    # MSC = SSC / dfc / nb_subjects

    # Sum square subject effect - between rows/subjects
    if SSC + SSE > SST:
        SSR = 0
    else:
        SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if round(MSR,2) == 0 or round(MSE,2) == 0:
        ICC = 0
    else:
        ICC = round((np.abs(MSR - MSE)),2) / round((MSR + dfc * MSE),2)

    return {label:ICC}
