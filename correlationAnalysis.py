from numpy import array,asarray,linspace,arange,iterable,rollaxis,delete,indices,nonzero,isnan,nan_to_num,all,any,round
from numpy.ma import corrcoef,mean,median,masked_array,sqrt,median,argsort,sum
from numpy.random import randint
from scipy.stats import norm

# ignore the future warning about ndarray == bool elementwise comparison
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# ------------------------------------------------------------------------------ #
# ----------------------- Sliding Window Functions------------------------------ #
# ------------------------------------------------------------------------------ #


# Sliding Window Iteration Object
def windowObject(sequence,winSize,step=1):
    
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]

# Sliding Window Function Returning Matrix of Windows in Columns
def slidingWindow(Signal,SamplingRate,WindowLength,Overlap) :
    
    '''Retruns windowed sequence in rows of matrix from a time series signal.
    Specify the window length and overlap in seconds. The sampling rate must be
    known as well.'''
    
    WindowIndex = int(SamplingRate*WindowLength); OverlapIndex = int(SamplingRate*Overlap)
    return array([ w for w in windowObject(Signal,WindowIndex,WindowIndex-OverlapIndex) ])

# ------------------------------------------------------------------------------ #
# ----------------------- Bootstrap Error Estimation --------------------------- #
# ------------------------------------------------------------------------------ #

def BCa( data, function = mean, alpha = 0.05, nBoot = 10000 ):
    
    """Given data array of shape (nSamples,*nVariables) and a statistics function
    that applies to that data, compute the bootstrap confidence interval using
    the bias-corrected and accelerated method [1].

    Parameters
    ----------
    
    data: array x = (nSamples,nFeatures), tuple of arrays ( x1, x2, ... )
    Input data or data tuple which corresponds to how many arguments the statistical
    function takes. Each data array can be one dimensional or two dimensional if the
    data points are vectors. In this case the function also has to accept vectors.
    
    function: function( *data ) -> value
    Statistical function on data, has to accept as many arguments f(x1,x2 ... ) such that
    the tuple, data = ( x1, x2, ... ) where each xi has the shape (nSamples,nFeatures)
    unless there is only one argument f(x) in which case, data = x
    
    alpha: float, default = 0.05
    Percentiles to use for the confidence interval, returned as (alpha/2, 1-alpha/2)
    
    nBoot : int, default = 100000
    Number of bootstrap samples to use in statistical estimations
    
    Returns
    -------
    
    confidences: tuple of floats
    The confidence percentiles specified by alpha 

    Examples
    --------

    Calculate the confidence intervals for the mean
    >> BCa( normal(size=100), mean )
    
    Calculate the confidence intervals for the covariance matrix
    >> BCa( (x,y), cov )

    References
    ----------
    
    [1] Efron, An Introduction to the Bootstrap. Chapman & Hall 1993, Section 14.3
    
    ----------"""

    # lower and upper confidence interval from alpha values
    alphas = array([alpha/2,1-alpha/2])

    # format data, account for multivariate functions f( x, ... )
    if isinstance(data, tuple) : data = tuple( array(x) for x in data )
    else : data = ( array(data), )
        
    # function applied to data
    stat = function(*data)
      
    # generate bootstrap and jackknife sample indexes
    bIndexes = bootstrapIndexes( data[0], nBoot )
    jIndexes = jackknifeIndexes( data[0] )
    
    # function applied to bootstrapped and jackknifed data
    bStat = array([ function(*(x[indexes] for x in data)) for indexes in bIndexes])
    bStat.sort(axis=0)
    
    jStat = array([ function(*(x[indexes] for x in data)) for indexes in jIndexes ])
    jMean = mean(jStat,axis=0)

    # bias correction and acceleration values
    z = norm.ppf( ( 1.0 * sum(bStat < stat, axis=0)  ) / nBoot )
    a = sum( (jMean - jStat)**3, axis=0 ) / ( 6.0 * sum( (jMean - jStat)**2, axis=0)**1.5 )
    
    # apply corrections
    zs = z + norm.ppf( alphas ).reshape( alphas.shape + z.ndim*(1,) )
    
    # compute intervals
    avals = norm.cdf( z + zs / (1-a*zs) )
    
    nvals = round((nBoot-1)*avals)
    nvals = nan_to_num(nvals).astype('int')

    if nvals.ndim == 1:
        
        # All nvals are the same. Simple broadcasting
        return bStat[nvals]
    
    else:
        # Nvals are different for each data point. Not simple broadcasting.
        # Each set of nvals along axis 0 corresponds to the data at the same
        # point in other axes.
        return bStat[(nvals, indices(nvals.shape)[1:].squeeze())]

def bootstrapIndexes( data, nBoot = 10000 ) :
    
    """Given data array of shape (nSamples,*nVariables) return a generator of
    length nBoot, where each element is an array of indexes of length nSamples.
    Resampling data uniformly with replacement gives a bootstrap sample.
    
    Parameters
    ----------
    
    data : array (nSamples,*nVariables)
    Input data where the first dimension is the number of samples
    
    nBoot : int, default = 100000
    Number of bootstrap samples to return
    
    Returns
    ----------
    
    out : iterator
    Generator object containing indexes of bootstrap samples
    
    ----------"""
    
    # get number of samples
    nSamples = len(data)
    
    # nBoot number of random resamplings with replacement
    for _ in xrange(nBoot) :
        yield randint(nSamples, size=nSamples)

def jackknifeIndexes( data ) :
    
    """Given data array of shape (nSamples,*nVariables) return a generator of
    length nSamples, where each element is an array of indexes of length
    nSamples-1. Removing one datum from data gives a jackknife sample.
    
    Parameters
    ----------
    
    data : array (nSamples,*nVariables)
    Input data where the first dimension is the number of samples
    
    Returns
    ----------
    
    out : iterator
    Generator object containing indexes of jackknife samples
    
    ----------"""
    
    # get number of samples
    nSamples = len(data)
    
    # yield all possible index arrays with one datum removed
    dataIndexes = arange(nSamples)
    return ( delete(dataIndexes,i) for i in dataIndexes )

# ------------------------------------------------------------------------------ #
# ----------------------- Pearson Cross Correlation ---------------------------- #
# ------------------------------------------------------------------------------ #

def pearsonCorrelation( X, Y, samplingRate, windowLength, nWindows=20, alpha=0.05, nBoot=None ) :
    
    # number of points, N and duration T of each series in ensembles X,Y
    N = len(X.T); T = N/samplingRate
    t = linspace(-T/2,T/2,N)
    
    # calculating overlap for given number of windows
    overlap = (nWindows*windowLength-N/samplingRate)/(nWindows-1)
    
    # construct array of delays
    tau = median(array(slidingWindow(t,samplingRate,windowLength,overlap)),axis=1)
    
    # windowing ensembles giving array of shape (nEnsemble,nWindows,nSamples)
    xWindows = array([ slidingWindow(x,samplingRate,windowLength,overlap) for x in X ])
    yWindows = array([ slidingWindow(y,samplingRate,windowLength,overlap) for y in Y ])
    
    # getting dimesions of windowed ensemble
    nEnsemble,nWindows,nSamples = xWindows.shape
    
    # for each stationary ergodic window, combine the samples across ensembles
    xWindows = rollaxis(xWindows,1).reshape(nWindows,nEnsemble*nSamples)
    yWindows = rollaxis(yWindows,1).reshape(nWindows,nEnsemble*nSamples)
    
    # dealing with missing data
    xWindows = masked_array(xWindows,isnan(xWindows))
    yWindows = masked_array(yWindows,isnan(yWindows))
    
    # 96% pearson coefficient confidence level 
    p = 1.96/sqrt(float(nEnsemble*nSamples))
    
    # pearson cross correlation coefficients for each window pair
    K = array([[ crosscorrcoef(x,y) for x in xWindows ] for y in yWindows ])
    
    # bootstrapped estimate of lower and upper confidence levels
    if nBoot != None :
        
        CI = array([[ BCa( (x,y), crosscorrcoef, alpha, nBoot ) for x in xWindows ] for y in yWindows ])
    
        # return temporal delays, pearson matrix, and confidence intervals
        return tau, K, CI[:,:,0], CI[:,:,1], p
    
    else :
        
        # return temporal delays, pearson matrix, and confidence intervals
        return tau, K, None, None, p

def crosscorrcoef(x,y) :
    
    """Take the numpy.ma.corrcoef function that deals with missing data,
    and automatically return cross correlation element in the matrix."""
    
    # return cross correlation
    return corrcoef(x,y)[0,1]

from mpl_toolkits.axisartist import Subplot
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.pyplot import figure

def plotCorrelation(tauArray,kappaMatrix,kappaLower=None,kappaUpper=None,CI=None,amplify=1):
    
    """Plots Pearson Correlation Coefficient K(t,tau) with rotated
    axis to indicate absolute t, and relative time shift tau, between
    two signals x(t),y(t).
    
    Specified matrix has to be square with values -1 < p < +1
    with corresponding time array giving the absolute time, t
    of the centers of each correlated window."""

    # defining tranformation for relative time shifts
    def R(x, y):
        x, y = asarray(x), asarray(y)
        #return x,y
        return (2*x - y)/2, (y + 2*x)/2

    def Rt(x, y):
        x, y = asarray(x), asarray(y)
        #return x,y
        return x + y, x - y

    # create figure with rotated axes
    fig = figure(figsize=(10, 10),frameon=False)
    grid_locator = angle_helper.LocatorDMS(20)
    grid_helper = GridHelperCurveLinear((R, Rt),
                  grid_locator1=grid_locator,
                  grid_locator2=grid_locator)
    
    ax = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)
    fig.add_subplot(ax);ax.axis('off');
    
    # copying over matrix
    K = array(kappaMatrix)
    
    # zero out correlations if confidence intervals overlap zero
    if all(kappaLower != None) and all(kappaUpper != None) :
        K[ (kappaLower<0) * (0<kappaUpper) ] = 0
        
    # zero out statistically insignificant correlations
    if all(CI != None) :
        K[ abs(kappaMatrix) < CI ] = 0
    
    # display pearson correlation matrix with +ive in red and -ive in blue
    ax.imshow(K,cmap="RdBu_r",interpolation="none",origin="bottom",
              extent = (tauArray[0],tauArray[-1],tauArray[0],tauArray[-1]),vmin=-1.0/amplify,vmax=1.0/amplify)

    # display rotated axes time,t and time delay,tau
    ax.axis["tau"] = tau = ax.new_floating_axis(0,0)
    ax.axis["t"] = t = ax.new_floating_axis(1,0)
    
    # setting axes options
    ax.set_xlim(tauArray[0],tauArray[-1])
    ax.set_ylim(tauArray[0],tauArray[-1])
    ax.grid(which="both")
    ax.set_aspect(1)
    
    return fig