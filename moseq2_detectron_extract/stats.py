import numpy as np


def is_outlier(points, thresh=3.5):
    ''' Returns a boolean array with True if points are outliers and False otherwise.

    Parameters:
        points (array): An numobservations by numdimensions array of observations
        thresh (float): The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
        mask (array): A numobservations-length boolean array.

    References:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    '''
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def exclude_outliers(data, threshold=3.5):
    ''' Returns the the values in data that are not considered outliers

    Parameters:
    points (array): An numobservations by numdimensions array of observations
    thresh (float): The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

    Returns:
    array: `data`, excluding outliers
    '''
    return data[(~is_outlier(data, threshold))]


def max_exclude_outliers(data, threshold=3.5):
    ''' Returns the max value in data, excluding outliers

    Parameters:
    points (array): An numobservations by numdimensions array of observations
    thresh (float): The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

    Returns:
    float: max value in `data`, excluding outliers
    '''
    return exclude_outliers(data, threshold).max()


def min_exclude_outliers(data, threshold=3.5):
    ''' Returns the min value in data, excluding outliers

    Parameters:
    points (array): An numobservations by numdimensions array of observations
    thresh (float): The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

    Returns:
    float: min value in `data`, excluding outliers
    '''
    return exclude_outliers(data, threshold).min()
