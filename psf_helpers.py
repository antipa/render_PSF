import numpy as np
import matplotlib.pyplot as plt
import copy
from IPython import display
from IPython.core.display import display as disp, HTML

def drawnow(fig):
    display.clear_output(wait=True)
    display.display(fig)
    

def genz(ii, rhomax):
    '''Generates Zernike polynomial function handles
    
     Inputs
     ii : Zernike index
     rhomax : max lateral coordinate, for normalization
    
     outputs:
     Z : surface height map
     fx: x gradient
     fy: y gradient
     These are handles meant to be evaluated on cartesian coordinates
    '''
    
    if ii == 1:
        Z = lambda x, y : np.ones_like(x)
        fx = lambda x, y : 0
        fy = lambda x, y : 0
    elif ii == 2:
        Z = lambda x, y : x/rhomax
        fx = lambda x, y : 1/rhomax
        fy = lambda x, y : 0
    elif ii == 3:
        Z = lambda x, y : y/rhomax
        fx = lambda x, y : 0
        fy = lambda x, y : 1/rhomax    
    elif ii == 4:
        Z = lambda x, y : (x/rhomax)**2 - (y/rhomax)**2
        fx = lambda x, y : 2*x/(rhomax**2)
        fy = lambda x, y : -2*y/(rhomax**2)
    elif ii == 5:
        Z = lambda x, y : -1 + 2*(x/rhomax)**2 + 2*(y/rhomax)**2
        fx = lambda x, y : 4*x/(rhomax**2)
        fy = lambda x, y : 4*y/(rhomax**2)
    elif ii == 6:
        Z = lambda x, y : 2*x*y/(rhomax**2)
        fx = lambda x, y : 2*y/(rhomax**2)
        fy = lambda x, y : 2*x/(rhomax**2)
    elif ii == 7:
        Z = lambda x, y : (x/rhomax)**3 - 3*(x/rhomax)*(y/rhomax)**2
        fx = lambda x, y : 3*x**2/(rhomax**3) - 3*y**2/(rhomax**3)
        fy = lambda x, y : 6*x*y/(rhomax**3)
    elif ii == 8:
        Z = lambda x, y : 3*(x/rhomax)**3 + 3*(x/rhomax)*(y/rhomax)**2 - 2*x/rhomax
        fx = lambda x, y : 9*x**2/(rhomax**3) + 3*y**2/(rhomax**3) - 2/rhomax
        fy = lambda x, y : 6*x*y/(rhomax**3)
    elif ii == 9:
        Z = lambda x, y : 3*(y/rhomax)**3 + 3*(x/rhomax)**2*(y/rhomax) - 2*y/rhomax
        fx = lambda x, y : 6*x*y/(rhomax**3)
        fy = lambda x, y : 9*y**2/(rhomax**3) + 3*x**2/(rhomax**3) - 2/rhomax
    elif ii == 10:
        Z = lambda x, y : -(y/rhomax)**3 + 3*(x/rhomax)**2*y/rhomax
        fx = lambda x, y : 6*x*y/(rhomax**3)
        fy = lambda x, y : -3*y**2/(rhomax**3) + 3*x**2/(rhomax**3)
    return Z, fx, fy




def genW2(zlist, coeffs, rhomax):
    '''
    Creates function handles to generate wavefronts as linear combinations 
     of zernike polynomials
    
    Inputs
    zlist : list of zernike indices
    coeffs : corresponding coefficients
    rhomax : maximum physical coordinate of the pupil, used to normalize
    
    Outputs
    W : Wavefront func handle, can be evaluated at 2D coordinates
    fx: x gradient handle
    fy: y gradient handle'''
    
    zern = []    
    for ii in range(len(zlist)):
        zern.append(genz(zlist[ii],rhomax))

    W = lambda x,y:sum([zern[ii][0](x,y)*coeffs[ii] for ii in range(len(zlist))])
    fx = lambda x,y:sum([zern[ii][1](x,y)*coeffs[ii] for ii in range(len(zlist))])
    fy = lambda x,y:sum([zern[ii][2](x,y)*coeffs[ii] for ii in range(len(zlist))])
    return W, fx, fy



def genSin(wx,wy,amp):
    '''
    Generates a sinusoidal phase map for modeling gratings.
    inputs
    wx : x-direction spatial frequency (in radians)
    wy : y-direction spatial frequency
    amp : amplitude of sinusoid
    
    Outputs
    W : Wavefront func handle, can be evaluated at 2D coordinates
    fx: x gradient handle
    fy: y gradient handle'''
    
    W = lambda x,y:amp*(np.sin(wx*x) + np.sin(wy*y))
    fx = lambda x,y:amp*wx*np.cos(wx*x)
    fy = lambda x,y:amp*wy*np.cos(wy*y)
    return W, fx, fy


def genSph(efl):
    '''
    Create function handles for spherical wavefront
    and derivatives, fx and fy.
    The curvature is designed so that it focuses at a deistance efl
    
    Inputs
    3fl : effective focal length of the surface (from vertex)
    
    Outputs
    W : Wavefront func handle, can be evaluated at 2D coordinates
    fx: x gradient handle
    fy: y gradient handle'''
    
    Wsph = lambda x,y : efl - efl*np.real(np.sqrt(1 - (x/efl)**2 - (y/efl)**2 + 0j))
    #fx = lambda x,y: x/np.sqrt(1 - (x/efl)**2 - (y/efl)**2)/efl
    #fy = lambda x,y: y/np.sqrt(1 - (x/efl)**2 - (y/efl)**2)/efl
    fx = lambda x,y: x/(efl * np.sqrt(1 - (y/efl)**2 - (x/efl)**2))
    fy = lambda x,y: y/(efl * np.sqrt(1 - (y/efl)**2 - (x/efl)**2))
    return Wsph, fx, fy


def trace_ray(rhox, rhoy, pupil,debug=False):
    '''
    Function that computes a ray's position and OPD at a flat plane,
    with the ray originating from an arbitrary wavefront.
    The ray direction and origin in z will be determined by the sum 
    of wavefront maps defined in the pupil.
    
    Inputs
    rhox : ray x coordinate
    rhoy : ray y coordinate
    pupil : dictionary containing pupil information.
    debug : print stuff if true
    
    Outputs
    xo and yo : ray output (x,y) coordinates
    opl : ray path length
    W : total wavefront (useful for analysis later)
    fx and fy : ray direction
    K : x direction cosine
    L : y direction cosine
    M : z direction cosine
    '''
    
    
    # Generate wavefront and gradients:
    W = pupil['W_sph'](rhox,rhoy) + pupil['W_aber'](rhox,rhoy) + pupil['W_sin'](rhox,rhoy)
    fx = pupil['fx_sph'](rhox,rhoy) + pupil['fx_aber'](rhox,rhoy) + pupil['fx_sin'](rhox,rhoy)
    fy = pupil['fy_sph'](rhox,rhoy) + pupil['fy_aber'](rhox,rhoy) + pupil['fy_sin'](rhox,rhoy)
    
    # z0 is the total propagton distance
    z0 = W - pupil['t']
    nlength = np.sqrt(1 + fx**2 + fy**2)   #Length of the wavefront normal vector
    
    # x,y, and z direction cosines, respectively
    K = -fx/nlength 
    L = -fy/nlength
    M = 1/nlength
    
    # Total propagation length along ray
    opl = -z0/M

    # Change in x and y ray coordinates
    dx = K*opl
    dy = L*opl
    
    # Output x and y coordinates
    
    yo = rhoy + dy
    xo = rhox + dx;
    
    #debugging stuff. 
    if debug:
        print('(rhox, rhoy):',rhox,rhoy)
        print('(Fx,Fy)',fx,fy)
        print('(dx, dy, dz)=',dx,dy,dz)        
        print('(xo,yo,zo)',xo,yo,z0)
        print('(K,L,M)=',K,L,M)
        print('Delta:', Delta)
        print('\n')

    

    return xo, yo, opl, W, fx, fy, K, L, M



def ray2wave(xo,yo,fx,fy,opl,lamb, Sx, Sy):
    '''    
    Generate plane wave based on ray data
     
    Inputs: 
    xo,yo : x,y location at plane
    fx,fy : x,y derivatives (slope of ray)
    opl : Ray optical path length
    lamb : Wavelength
    Sx, Sy : meshgrided coordinates of sensor pixels
    
    Outputs
    U : Complex field of plane wave evaluated on sensor grid
    '''
    
    k = 2*np.pi/lamb
    opd = k*opl
    nnorm = np.sqrt(fx**2 + fy**2 + 1)
    kx = -fx*k/nnorm
    ky = -fy*k/nnorm
#    kz = np.sqrt(k**2 - kx**2 - ky **2)

    
    U = np.exp(1j*(-kx*(Sx-xo) - ky*(Sy-yo) - opd))
    return U


def huygens(pupil, sensor, Nrays, lam, rx, ry, disp_interval = 100):
    # Plane waves loop
    
    xo = []
    yo = []
    opl = []
    fx = []
    fy = []
    dx = []
    dy = []

    U_accum = np.zeros(sensor['npix']) + 1j*np.zeros(sensor['npix'])   #This will accumulate the total field
    f = plt.figure(figsize=(20,20))   #Figure handle

    # Generate sensor pixel grid
    sx = np.linspace(-sensor['fov'][1]/2,sensor['fov'][1]/2,sensor['npix'][1])
    sy = np.linspace(-sensor['fov'][0]/2,sensor['fov'][0]/2,sensor['npix'][0])
    Sx,Sy = np.meshgrid(sx,sy)

    # Loop that randomly samples rays
    for kk in range(Nrays):


        # Pick a random location on the surface
        if pupil['ap_type'] == 'circ':
            # Generate rays randomly in polar coordinates
            thx = np.random.rand(1)*2*np.pi     #Pick random ray azimuth
            rho = pupil['ca'][0]/2*np.sqrt(np.random.rand(1))   #Random ray radial location

            #Convert to cartesian
            rhox = rho*np.sin(thx)
            rhoy = rho*np.cos(thx)

        elif pupil['ap_type'] == 'rect':
            rhox = pupil['ca'][1]*(np.random.rand(1) - 0.5)
            rhoy = pupil['ca'][0]*(np.random.rand(1) - 0.5)

        rhox = np.clip(rhox,rx[0],rx[-1])
        
        rhoy = np.clip(rhoy,ry[0],ry[-1])
        # Launch a ray toward the sensor
        rays_out = trace_ray(rhox, rhoy, pupil,debug=False)
        # This part "remembers" ray data. Commented out because not needed
    #     xo.append(rays_out[0])
    #     yo.append(rays_out[1])
    #     dx.append(xo[kk] - rhox)
    #     dy.append(yo[kk] - rhoy)
    #     opl.append(rays_out[2])
    #     fx.append(rays_out[4])
    #     fy.append(rays_out[5])
        xo_d = rays_out[0]
        yo_d = rays_out[1]
        fx_d = rays_out[4]
        fy_d = rays_out[5]
        opl_d = rays_out[2]

        U = ray2wave(xo_d, yo_d, fx_d, fy_d, opl_d, lam, Sx, Sy)

        U_accum += U
        if kk%100==0:
            plt.imshow(np.abs(U_accum)**2)#,extent=figextent)
            #plt.imshow(np.real(U))
            plt.title(kk)
            print(kk)
            drawnow(f)
        # Create the corresponding plane wave on the sensor grid

        # Add this plane wave to the existing distribution

        # Repeat

#    U_out = U_accum
    return U_accum



# Compare to Fresnel propagation
def propagate_field(lam, ps, z, U):
    k = 2*np.pi/lam   
    siz = np.shape(U)
    fx = np.linspace(-1/2/ps,1/2/ps,siz[1])
    fy = np.linspace(-1/2/ps,1/2/ps,siz[0])
    x = np.linspace(-siz[1]/2*ps,siz[1]/2*ps,siz[1])
    y = np.linspace(-siz[0]/2*ps,siz[0]/2*ps,siz[0])
    X,Y = np.meshgrid(x,y)
    Fx,Fy = np.meshgrid(fx,fy)
    Uf = np.fft.fftshift(np.fft.fft2(U))
    Hf = np.exp(1j*2*np.pi*z/lam * np.sqrt(0j+1-(lam*Fx)**2 - (lam*Fy)**2))
    Up = np.fft.ifft2(np.fft.ifftshift(Uf*Hf))
    return Up

def rays_to_im(xo, yo, sensor):
    '''
    convert 2D ray coordinates to binned sensor image. 
    Inputs:
    xo: list of x coordinates of rays
    yo: list of y coordinates of rays
    sensor: sensor dict made using examples of previous code. Sorry, I don't have a nice class yet...

    Outputs
    H : the image after binning
    xe : bin edges x
    ye: bin edges y
    '''
    
    yoarr = np.asarray(yo)[:,0]
    xoarr = np.asarray(xo)[:,0]
    binedgex = np.r_[:sensor['npix'][1]+1]*sensor['px']
    binedgex -= np.mean(binedgex)
    binedgey = np.r_[:sensor['npix'][0]+1]*sensor['px']
    binedgey -= np.mean(binedgey)
    H,xe, ye = np.histogram2d(xoarr,yoarr,
                              bins=[sensor['npix'][1],sensor['npix'][0]],
                              range=[[-sensor['fov'][1]/2,sensor['fov'][1]/2],[-sensor['fov'][0]/2,sensor['fov'][0]/2]])
    return H, xe, ye