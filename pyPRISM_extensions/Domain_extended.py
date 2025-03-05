#!python
'''This code extends "pyPRISM" framework to 2 spatial dimensions. 
   This file should be called in substitution for '~pyPRISM/core/Domain.py'.
   
   Fourier transform for radially symmetric functions in 2d reduces to Hankel transform,
   which is efficiently computed 'fast' by "quasi discrete Hankel transform (QDHT)" 
   and is implemented using "PyHank" [`link <https://github.com/etfrogers/pyhank>`__]
   
   References
   ----------
   [1] Yu, L., M. Huang, M. Chen, W. Chen, W. Huang, and Z. Zhu (1998)
        "Quasi-discrete Hankel transform", 
        Optics Letters 23, 6.; [`link
        <https://opg.optica.org/ol/fulltext.cfm?uri=ol-23-6-409&id=36623>`__]
        
   [2] Guizar-Sicairos, M. and J. C. Gutiérrez-Vega (2004)
        "Computation of quasi-discrete Hankel transforms of integer order for propagating optical wave fields",
        J. Opt. Soc. Am. A 21 1.; [`link
        <https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-21-1-53&id=78247>`__]    
        
    Changes from original Domain.py
    -------------------------------
    - argument 'dim' is added to function '__init__', which is set to '3' by default
    - functions 'hankel', 'to_fourier_2d', and 'ro_real_2d' are added, which are based on ''HankelTransform' from "pyhank"
    - functions 'MatrixArray_to_fourier' and 'MatrixArray_to_real' are modified to support the case of dim==2
 
'''

from __future__ import division,print_function
from pyPRISM.core.Space import Space
import numpy as np
from scipy.fftpack import dst
from pyhank import HankelTransform

class Domain(object):
    r'''Define domain and transform between Real and Fourier space

    **Mathematical Definition**
        
        The continuous, 1-D, radially symmetric Fourier transform is written
        as follows:

        .. math::

            k\ \hat{f}(k) = 4 \pi \int r\ f(r) \sin(k\ r) dr (in 3d)
            k\ \hat{f}(k) = 2 \pi \int r\ f(r) \J0(k\ r) dr (in 2d; J0 is 0th-order Bessel function of the first kind)

        We define the following discretizations

        .. math:: 

            r = (i+1)\Delta r

            k = (j+1)\Delta k

            \Delta k = \frac{\pi}{\Delta r (N + 1)}

        to yield

        .. math::

            \hat{F}_j = 4 \pi \Delta r \sum_{i=0}^{N-1} F_i \sin\left(\frac{\pi}{N+1} (i+1)(j+1)\right) (in 3d)
            \hat{F}_j = 2 \pi \Delta r \sum_{i=0}^{N-1} F_i  \J0\left(\frac{\pi}{N+1} (i+1)(j+1)\right) (in 2d)

        with the following definitions:

        .. math::

            \hat{F}_j = (j+1)\ \Delta k\ \hat{f}((j+1)\Delta k) = k \hat{f}(k)

        .. math::

            F_i = (i+1)\Delta r\ f((i+1)\Delta r) = r f(r)

        The above equations describe a Real to Real, type-I discrete sine
        transform (DST) in 3d. To tranform to and from Fourier space we will use 
        the type-II and type-III DST's respectively in 3d. With Scipy's interface to
        fftpack, the following functional coeffcients 

        .. math::

            C^{DSTII} = 2 \pi r \Delta r (in 3d)

        .. math::

            C^{DSTIII} = \frac{k \Delta k}{4 \pi^2} (in 3d)
    
    **Description**

        Domain describes the discretization of Real and Fourier space
        and also sets up the functions and coefficients for transforming
        data between them.
    
    '''
    def __init__(self,length,dim=3,dr=None,dk=None):
        r'''Constructor

        Arguments
        ---------
        length: int
            Number of gridpoints in Real and Fourier space grid

        dr,dk: float
            Grid spacing in Real space or Fourier space. Only one can be
            specified as it fixes the other.
            
        dim: int
            Spatial dimension (must be 2 or 3)

        '''
        assert (dim==3) or (dim==2), "DIMENSION MUST BE 3 or 2!!!"
        
        self._length = length
        self.dim = dim
        
        if (dr is None) and (dk is None):
            raise ValueError('Real or Fourier grid spacing must be specified')
            
        elif (dr is not None) and (dk is not None):
            raise ValueError('Cannot specify **both** Real and Fourier grid spacings independently.')
            
        elif dr is not None:
            self.dr = dr #dk is set in property setter
            
        elif dk is not None:
            self.dk = dk #dr is set in property setter          
            
        self.build_grid() #build grid should have been called already but we'll be safe
        
        if dim == 2:
            self.hankel() #build hankel transformation
    
    def build_grid(self):
        '''Construct the Real and Fourier Space grids and transform coefficients'''
        self.r = np.arange(self._dr,self._dr*(self._length+1),self._dr)
        self.k = np.arange(self.dk,self.dk*(self._length+1),self.dk)
        self.DST_II_coeffs = 2.0*np.pi *self.r*self._dr 
        self.DST_III_coeffs = self.k * self.dk/(4.0*np.pi*np.pi)
        self.long_r = self.r.reshape((-1,1,1))
    
    @property
    def dr(self):
        '''Real grid spacing'''
        return self._dr
    @dr.setter
    def dr(self,value):
        self._dr = value
        self._dk = np.pi/(self._dr*self._length)
        self.build_grid()#need to re-build grid since spacing has changed
    
    @property
    def dk(self):
        '''Fourier grid spacing'''
        return self._dk
    @dk.setter
    def dk(self,value):
        self._dk = value
        self._dr = np.pi/(self._dk*self._length)
        self.build_grid()#need to re-build grid since spacing has changed
        
    @property
    def length(self):
        '''Number of points in grid'''
        return self._length
    @length.setter
    def length(self,value):
        self._length = value
        self.build_grid()#need to re-build grid since length has changed
        
    def __repr__(self):
        return '<Domain length:{} dr/rmax:{:4.3f}/{:3.1f} dk/kmax:{:4.3f}/{:3.1f} dim:{:d}>'.format(self.length,self.dr,self.r[-1],self.dk,self.k[-1],self.dim)
    
    def to_fourier(self,array):
        r''' Discrete Sine Transform of a numpy array 
        
        Arguments
        ---------
        array: float ndarray
            Real-space data to be transformed
            
        Returns
        -------
        array: float ndarray
            data transformed to fourier space


        Peforms a Real-to-Real Discrete Sine Transform  of type II 
        on a numpy array of non-complex values. For radial data that is 
        symmetric in :math:`\phi` and :math`\theta`, this is a correct transform
        to go from Real-space to Fourier-space. 
        
        
        '''
        return dst(self.DST_II_coeffs*array,type=2)/self.k
    
    def to_real(self,array):
        ''' Discrete Sine Transform of a numpy array 
        
        Arguments
        ---------
        array: float ndarray
            Fourier-space data to be transformed
            
        Returns
        -------
        array: float ndarray
            data transformed to Real space

        Peforms a Real-to-Real Discrete Sine Transform  of type III
        on a numpy array of non-complex values. For radial data that is 
        symmetric in :math:`\phi` and :math`\theta`, this is a correct transform
        to go from Real-space to Fourier-space. 
        
        '''
        return dst(self.DST_III_coeffs*array,type=3)/self.r
    
    def MatrixArray_to_fourier(self,marray):
        ''' Transform all pair-functions of a MatrixArray to Fourier space in-place

        Arguments
        ---------
        marray: :class:`pyPRISM.core.MatrixArray.MatrixArray`
            MatrixArray to be transformed

        Raises
        ------
        *ValueError*:
            If the supplied MatrixArray is already in Real-space
        '''
        assert (self.dim==3) or (self.dim==2), "DIMENSION MUST BE 3 or 2!!!"
        
        if marray.space == Space.Fourier:
            raise ValueError('MatrixArray is marked as already in Fourier space')
           
        for (i,j),(t1,t2),pair in marray.iterpairs():
                marray[t1,t2] = self.to_fourier(pair) if (self.dim==3) else self.to_fourier_2d(pair)
   
        marray.space = Space.Fourier
            
    def MatrixArray_to_real(self,marray):
        ''' Transform all pair-functions of a MatrixArray to Real space in-place 

        Arguments
        ---------
        marray: :class:`pyPRISM.core.MatrixArray.MatrixArray`
            MatrixArray to be transformed

        Raises
        ------
        ValueError:
            If the supplied MatrixArray is already in Real-space
        '''
        assert (self.dim==3) or (self.dim==2), "DIMENSION MUST BE 3 or 2!!!"
        
        if marray.space == Space.Real:
            raise ValueError('MatrixArray is marked as already in Real space')
            
        for (i,j),(t1,t2),pair in marray.iterpairs():
            marray[t1,t2] = self.to_real(pair) if (self.dim==3) else self.to_real_2d(pair)
            
        marray.space = Space.Real
        
    def hankel(self):
        '''define hankel transform (Hk -> Hr as dr,dk -> 0)'''
        Hr = HankelTransform(order=0, radial_grid=self.r)
        Hk = HankelTransform(order=0, k_grid=self.k)
        self._Hr = Hr
        self._Hk = Hk
            
    def to_fourier_2d(self,array):
        '''Quasi Discrete Hankel Transform of a numpy array 
        
        Arguments
        ---------
        array: float ndarray
            Real-space data to be transformed
            
        Returns
        -------
        array: float ndarray
            data transformed to fourier space

        Peforms a Real-to-Real Quasi Discrete Hankel Transform
        on a numpy array of non-complex values. For radial data that is 
        symmetric in :math`\theta`, this is a correct transform
        to go from Real-space to Fourier-space. 
        '''
        # adjust array's spacing to that of the optimized grid H.r
        array_adjusted = self._Hr.to_transform_r(array) 
        # perform QDHT
        array_adjusted_transformed = self._Hr.qdht(array_adjusted)
        # adjust array's spacing back to that of the original grid self.r
        array_transformed = self._Hr.to_original_r(array_adjusted_transformed) 
        return array_transformed
    
    def to_real_2d(self,array):
        ''' Discrete Sine Transform of a numpy array 
        
        Arguments
        ---------
        array: float ndarray
            Fourier-space data to be transformed
            
        Returns
        -------
        array: float ndarray
            data transformed to Real space

        Peforms a Real-to-Real Inverse Quasi Discrete Hankel Transform
        on a numpy array of non-complex values. For radial data that is 
        symmetric in :math`\theta`, this is a correct transform
        to go from Real-space to Fourier-space. 
        
        '''
        # adjust array's spacing to that of the optimized grid H.kr
        array_adjusted = self._Hk.to_transform_k(array)
        # perform inverse QDHT
        array_adjusted_transformed = self._Hk.iqdht(array_adjusted)
        # adjust array's spacing back to that of the original grid self.k
        array_transformed = self._Hk.to_original_k(array_adjusted_transformed) 
        return array_transformed
      
    
