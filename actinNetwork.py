"""
The following code was created during a Rotation in Sixt Group, in an effort
to reproduce electron microscopy data provided by Jan Muller. Thank you to all
who participated in fruitful discussions. Don't hesitate to contact me for
advice on implementation.

Szep, G. 2015. Institute of Science and Technology Austria
email: gregory.szep@ist.ac.at

25/03/16 - Distinguishing branches, caps and filament monomers

Restructuring of the code to include the option of recording not only positions
but also which type of protien at that location.

14/03/16 - Growth control upgrade / Bugfix

Major bugfix involving incorrectly used pointer arithmetic resulted in network
stability towards oscillations and noise. Thanks to the advice of Srdjan Sarikas
from the institute, the model more accuratly represents the processes driving
the cytoskeleton.

10/03/16 - Growth control upgrade

Included inverse proportionality of branching probability wrt actin density.
This way the growth of the network is limited by the availability of 
nuleation promotingfactors.

17/12/15 - Compatible with Python 3.3 and Mac OSX Yosemite 10.10.5

"""

# ------------------------------- pre-requisistes ------------------------------------- #

# import libraries
from numpy import array,matrix,arange,cumsum,linspace,append,gradient,delete,all,in1d,histogram,around,amax,amin,mean,cos,sin,cosh,arctan,arctan2,exp,log,pi,meshgrid,median,argmax,NaN
from scipy.stats import kurtosis
from numpy.random import poisson,normal,randint,uniform,binomial
from numpy.linalg import norm
from scipy.signal import savgol_filter
from matplotlib.pyplot import figure,subplots,savefig,close,plot,subplot,axes,axis,xlim,xlabel,ylim,ylabel,legend,contourf
from os import system,getcwd

# ------------------------------- function definitions ------------------------------------- #

# defining heavside pi function, width dx
def heavisidePi(x,dx):
    
    """Return 1.0 if abs(x) <= dx/2.0, otherwise 0.0. This is
    also known as a unit hat function with width, dx. """
    
    return 1.0 if abs(x) <= dx/2.0 else 0.0

# defining heavside theta function
def heavisideTheta(x):
    
    """Returns 1.0 if 0 <= x and otherwise 0.0, otherwise known
    as the unit step function."""
    
    return 1.0 if 0 <= x else 0.0

# sampling function from bimodal distribution
def bimodal(dx,sigma,nSize=1):

    """Bimodal distribution is defined as a normalised superposition
    of Gaussians each of standard deviation sigma, centered at +dx,-dx.
    This function returns an array of real numbers sampled from it."""

    # random choice of mode
    choices = randint(2,size=nSize)

    # generate distribution
    xDist = array([ normal(loc=-dx, scale=sigma) if i == 0 
               else normal(loc= dx, scale=sigma) for i in choices ])

    return xDist if nSize > 1 else xDist[0]

# function that returns the mode of a list of values
def mode(xArray,binRes=100) :
    
    """This function returns the mode, as the center of the bin with
    maximum hitorgram value of xArray, given number of bins, binRes."""
    
    # generate histogram
    hist, bins = histogram(xArray,bins=binRes)
    
    # get bin centers
    centers = (bins[:-1] + bins[1:]) / 2.0
    
    # return mode of histogram
    return centers[argmax(hist)]

# ------------------------------- main class object ------------------------------------- #

# class object, for initialising and running simulation
class network(object) :
    
    """Object that defines a frontier of binding sites in two dimensions
    initialised by list of tuples (xSeed,dxSeed) where x,dx gives the position
    and orientation of the binding sites respectively. Periodic boundary
    conditions, given by the range of xSeed, in the x-direction is applied.
    
        The user must also specify space [x,y], time t, and orientation
    phi, dependent functions rLambda,rBeta,rKappa which must take on the
    form f(x,y,t). These functions give the rate of elongation, branching
    and capping at the binding sites.
    
        In addition the user may specify the branching statistics, branchTheta
    and branchSigma which give the mean and standard deviation of a gaussian
    mixture distribution, centered at +branchTheta, -branchTheta. The direction
    of branching is forced, by default but can be switched off.
    
    Szep, G. 2015. Institute of Science and Technology Austria"""
    
    # initialisation
    def __init__(n,rLambda,rBeta,rKappa,xSeed,dxSeed,
                 branchTheta=1.3003,branchSigma=0.1354,
                 forceDirection=True,recordHistory=False):
        
        # defining rate functions of space, (x,y) and time, t
        n.rLambda = rLambda; n.rBeta = rBeta; n.rKappa = rKappa
        
        # set branching statistics
        n.branchTheta = branchTheta; n.branchSigma = branchSigma
        
        # prepare for periodic boundary condition along x-axis
        xSeed.T[0] -= amin(xSeed.T[0]); n.xBoundary = amax(xSeed.T[0])
        
        # inital network frontier as list of binding sites [ ... (x,dx) ... ]
        n.Frontier = zip(xSeed,dxSeed)
        n.monomerSize = norm(dxSeed[0])
        
        # initialise number of events, mode angle
        n.nCapped = 0; n.nBranched = 0; n.nBarbed = len(xSeed)
        n.phiMax = abs( mode( n.getAngles(n.Frontier) ) )
        n.nFilaments = [[ n.nBarbed, n.nBranched, n.nCapped, n.phiMax ]]
        
        # position of leading edge and time elapsed
        n.tElapsed = 0.0
        n.xEdge = 0.0
        
        # whether to force growth direction and record history
        n.forceDirection = forceDirection
        n.recordHistory = recordHistory
        
        # if we record history in these data arrays, to record all (x,dx)
            
        n.Monomers = zip(xSeed,dxSeed)
        n.Branches = []
        n.Caps = []
        
    # function that makes two dimensional vector, periodic in x-axis
    def xPeriodic(n,rArray) :
        
        """Applying periodic boundary conditions to two dimensional
        position vectors given by rArray, in the x-direction. The
        boundary edge specified by the largest x-value of xSeed."""

        # vector components
        x = rArray[0]; y = rArray[1]

        # periodic in x-axis
        x %= n.xBoundary

        return array([x,y])
    
    # extract positions out of frontier
    def getPositions(n,sites) :
        
        """This function returns an array of positions of the network
        frontier which has the form of a list tuples with position
        and displacement vector (x,dx) for each binding site."""
        
        return array([ x for x,dx in sites ])
    
    # extract angles out of frontier
    def getAngles(n,sites) :
        
        """This function returns an array of angles on the network
        frontier which has the form of a list tuples with position
        and displacement vector (x,dx) for each binding site."""
        
        return array([ arctan2(dx[0],dx[1]) for x,dx in sites ])
    
    # function that elongates site by monomer
    def elongate(n,iIndex) :
        
        """This function elongates ith site by the magnitude of the
        orientation vector |dr|. This elongated site replaces the
        previous position in the frontier."""
    
        # extract positon and direction
        r,dr = n.Frontier[iIndex]

        # elongates site by dr, obeying periodic boundary
        n.Frontier[iIndex] = ( n.xPeriodic(r+dr), dr )
        
        # optional record of history
        if n.recordHistory == True : n.Monomers += [ ( n.xPeriodic(r+dr), dr ) ]
        
        
    # function that adds site at fixed angle, random chirality with respect to site
    def branch(n,iIndex):
        
        """This function adds a binding site at the same position as
        ith site, with an orientation rotated at a random angle theta. This
        angle is sampled from a bimodal distribution, given by branchTheta
        and branchSigma.
        
            By default, filaments created at orientations beyond +pi/2,-pi/2
        are rejected, so that growth only occurs in one direction. This
        option is changed by setting forceDirection = False. """
        
        # extract positon and direction
        r,dr = n.Frontier[iIndex]
        
        # direction
        x = dr[0]; y = dr[1]
        
        if n.forceDirection == True :
            while True :
            
                # sampling angle 
                theta = bimodal(dx=n.branchTheta,sigma=n.branchSigma)

                # rotate direction by sampled angle
                dr = array([ x*cos(theta)-y*sin(theta) , x*sin(theta)+y*cos(theta) ])
                
                # only accept directed branches
                if dr[1] > 0 : break
        
        else :
            
            # sampling angle 
            theta = bimodal(dx=n.branchTheta,sigma=n.branchSigma)

            # rotate direction by sampled angle
            dr = array([ x*cos(theta)-y*sin(theta) , x*sin(theta)+y*cos(theta) ])
                
        # add binding site
        n.Frontier += [ (r, dr) ]
        n.nBranched += 1; n.nBarbed += 1
        
        # optional record of history
        if n.recordHistory == True : n.Branches += [ (r, dr) ]
        
    # function that removes ith binding site from frontier
    def cap(n,iIndex) :
        
        """This function removes ith site from Frontier, and updates
        statistics of capped filaments."""
        
        # extract positon and direction
        r,dr = n.Frontier[iIndex]

        # remove ith binding site
        del n.Frontier[iIndex]
        n.nCapped += 1; n.nBarbed -= 1
        
        # optional record of history
        if n.recordHistory == True : n.Caps += [ (r, dr) ]

    # evolution during step size dt
    def timeStep(n,dt,Fext=0.0) :
        
        """This function iterates through all the sites in the frontier
        generating elongation, branching and capping events from poisson
        distributions in a small time step dt, with rates, rLambda, rBeta,
        rKappa respectively."""
        
        # initialising iteration through frontier
        tFrontier = list(n.Frontier); i = 0
        
        # denisty of barbed ends now
        n.D = n.nBarbed/n.xBoundary
        
        # for all binding sites events arrive according to poisson proccess
        for x,dx in tFrontier :
            
            # capping removes binding site
            capping = poisson( n.rKappa( x[0], x[1] - n.xEdge, n.tElapsed ) * dt )
            if bool(capping) == True :
                n.cap( i )
                continue 
                
            # branching creates new binding sites
            branching = poisson( n.rBeta( x[0], x[1] - n.xEdge, n.tElapsed) * dt  )
            if bool(branching) == True :
                n.branch( i )
                
            # elongation by n monomer lengths
            elongating = poisson( n.rLambda( x[0], x[1] - n.xEdge, n.tElapsed ) * dt )
            if bool(elongating) == True :
                n.elongate( i )
                
            # incrementor
            i += 1
                
        # evolve time 
        n.tElapsed += dt
        
        # velocity of membrane due to brownian ratchet
        vo = n.rLambda( 0.0, 0.0, 0.0 )*n.monomerSize
        kT = 4.1
        
        n.v = vo*exp(-Fext*n.monomerSize/(kT*n.D))
        n.xEdge += n.v*dt
        
        
        
        #DiffEdge = 0.1
        #n.v = normal(scale=2.0*DiffEdge/dt)
        #dx = n.v*dt
        
        # resulting shift of polymerising region
        #if n.xEdge + dx > Xmax :
        #    n.xEdge += dx
        #elif n.xEdge + dx < Xmax :
        #    n.xEdge = 2*Xmax - (n.xEdge + dx)
            
    
    # evolve until time, tFinal, with option of recording evolution data
    def evolve(n,dt,tFinal,Fext=0.0) :
        
        """This function evolves the frontier to time, tFinal, in increments
        dt, with the option of recording all (x,dx) at each time step, which
        by default recordHistory = False."""

        # while networks grows and up until time tFinal
        while n.nBarbed != 0 and n.tElapsed <= tFinal :
            
            # evolve
            n.timeStep(dt,Fext)
            
            # increment counting statistics
            n.phiMax = abs( mode( n.getAngles(n.Frontier) ) )
            n.nFilaments += [[ n.nBarbed, n.nBranched, n.nCapped, n.phiMax ]]
    
    # export positions and angles as animation, up until time tFinal
    def exportData(n,dt,ds,tFinal,Fext=0.0) :
        
        """This function runs the simulation up until time, tFinal in
        increments dt, and exports statistics in increments ds. The
        output gives animations showing evolution of angular distribution
        plot of the filaments in space, and population of filaments."""

        # initialise index
        j = 1
        
        # while networks grows and up until time tFinal
        while n.nBarbed != 0 and n.tElapsed <= tFinal :

            # create save and close figures 
            n.plotAngles().savefig("figures/angles"+str(j).zfill(3)+".png"); close();
            if n.recordHistory == True : 
                n.plotData().savefig("figures/network"+str(j).zfill(3)+".png"); close();
                
            # generate data for window ds, in step size, dt
            n.evolve( dt, n.tElapsed + ds , Fext)
            
            # inrementation
            j+=1
        
        # export filament population
        # n.plotFilaments().savefig("output/filamentStatistics.png"); close();
        
        # convert saved figures into animated gif then delete cache
        system("convert -delay 10 -loop 0 figures/angles*.png output/anglesDistribution.gif");
        system("rm -R figures/angles*.png");
        if n.recordHistory == True :
            system("convert -delay 10 -loop 0 figures/network*.png output/networkPlot.gif");
            system("rm -R figures/network*.png");
    
    # plotting network positions
    def plotData(n) :
        
        """This function returns a plot of filaments in space, superposed on
        top of a contour plot of the elongation rate rLambda. This plot reads
        out configuration of the network at the current time step."""
        
        
        # getting positions
        xFil = n.getPositions( n.Monomers )
        xBranch = n.getPositions( n.Branches )
        xCap = n.getPositions( n.Caps )

        # create coordinate mesh
        yRange = arange(0.0,n.xBoundary); xRange = arange(n.xEdge+30.0-n.xBoundary,n.xEdge+30.0)
        xGrid, yGrid = meshgrid(xRange, yRange)
        zGrid = array([[ n.rLambda(Y,X-n.xEdge,0.0) for X in xRange] for Y in yRange])

        # figure plotting actin network
        Plot = figure(figsize=(16,16))

        # plotting all monomers as different colour points 
        plot(xFil.T[1],xFil.T[0],'g',marker=".",linewidth=0,ms=5.5,alpha=0.5)
        plot(xBranch.T[1],xBranch.T[0],'#2737ff',marker=".",linewidth=0,ms=10)
        plot(xCap.T[1],xCap.T[0],'#ff0000',marker=".",linewidth=0,ms=10)

        # plotting options
        xlabel(r"Distance, $x$ / nm",fontsize=28)
        ylabel(r"Distance, $y$ / nm",fontsize=28)
        xlim(n.xEdge+30.0-n.xBoundary,n.xEdge+30.0)
        ylim(0,n.xBoundary)
        axes().set_aspect('equal', 'box')
        axes().tick_params(labelsize=16)

        # together with rate envelope
        contourf(xGrid, yGrid, zGrid, cmap='Greens',vmin=140.0,vmax=141);
        
        # return figures
        return Plot
    
    def plotFilaments(n,smoothingWindow=11,smoothingOrder=3) :
        
        """This plot returns the population of barbed ends, the rate of change
        of capped and branched ends, normalised to be viewed in the same scale,
        where 1.0 represents the maximum population acheived over the whole
        simulation."""
        
        # extract filament numbers
        barbed = (array(n.nFilaments).T[0]).astype(float)
        branched = gradient(array(n.nFilaments).T[1]).astype(float)
        capped = gradient(array(n.nFilaments).T[2]).astype(float)
        
        # extract filament mode angle
        phiMax = (array(n.nFilaments).T[3]).astype(float)

        # normalise to variation
        barbed /= mean(barbed)
        branched /= mean(branched)
        capped /= mean(capped)

        # maximum
        yMax = max(amax(barbed),amax(branched),amax(capped))

        # time range
        t = linspace( 0.0, n.tElapsed, num=len(barbed) )

        # plotting number of filaments
        f,ax1 = subplots()

        ax1.plot(t,savgol_filter(barbed,smoothingWindow,smoothingOrder),'g',label="Barbed Ends")
        ax1.plot(t,barbed,'g,')
        ax1.plot(t,savgol_filter(branched,smoothingWindow,smoothingOrder),'b',label="Branching Rate")
        ax1.plot(t,branched,'b,')
        ax1.plot(t,savgol_filter(capped,smoothingWindow,smoothingOrder),'r',label="Capping Rate")
        ax1.plot(t,capped,'r,')
        legend();

        ax2 = ax1.twinx()
        ax2.plot(t,savgol_filter(180*phiMax/pi,smoothingWindow,smoothingOrder),'k')
        ax2.plot(t,180*phiMax/pi,'k,')


        # plot labels
        ax1.set_xlabel(r"Time, $t$ / $\sec$", fontsize=16);
        ax1.set_ylabel(r"Count Variation, $n(t)$", fontsize=16)
        ax2.set_ylabel(r"Mode Filament Angle, $\phi(t)$ / $\deg$", fontsize=16)
        xlim(0,n.tElapsed);

        f.set_figheight(6); f.set_figwidth(12)
        
        return f
    
    # plot angular distribution
    def plotAngles(n) :
        
        """Returns a polar plot of the normalised angular histogram, of
        orientation of binding sites at the frontier of the network."""
        
        # creating figure and histogram
        angleHistogram = figure(figsize=(8,8))
        hist, bins = histogram( n.getAngles( n.Frontier ), bins=40,normed=False)
        centers = (bins[1:]+bins[:-1])/2.0

        # polar plot
        ax = subplot(111,projection="polar")
        ax.bar(centers, hist, color='g', width= 2*pi/40,edgecolor="none",align="center")
        ax.tick_params(labelsize=16)

        # labels
        ax.set_xlabel(r"Elongating Filament Angles, $\phi$ / $^{o}$", fontsize=28)

        ax.set_theta_zero_location('E')
        ax.set_theta_direction('clockwise')
        ax.set_ylim(0,25)
        ax.set_yticks(array([0]))
        ax.set_xticks(array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,NaN,180])/180*pi);
        
        return angleHistogram
