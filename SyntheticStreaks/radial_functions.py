#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:29:10 2023

@author: Akalanka
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import skimage.measure
import pylab
from skimage.filters import difference_of_gaussians, window, gaussian
from scipy.optimize import curve_fit
from scipy import interpolate
from skimage.draw import disk
import math
from scipy.stats import multivariate_normal
from scipy.stats import moment
#import cv2

class gaussian_blob:

    def __init__(self,amplitude,mean,spread,num):
        self.mean = mean
        self.sigma = spread
        self.A = amplitude
        self.num =num
        
    def mesh(self):
        x = np.linspace( self.mean[0]-8*self.sigma[0], self.mean[0]+8*self.sigma[0], num=self.num)
        y = np.linspace(self.mean[1]-8*self.sigma[1], self.mean[1]+8*self.sigma[1], num=self.num)
        X, Y = np.meshgrid(x,y)

        return X, Y

        
    def static_gauss(self):

        COV = [ [self.sigma[0]**2, 0], [0, self.sigma[1]**2] ] # currently un-correlated variables

        X , Y = self.mesh() 
        x_ = X.flatten()
        y_ = Y.flatten()
        xy = np.vstack((x_, y_)).T
        
        #gaussian build
        normal_rv = multivariate_normal(self.mean, COV)
        z = normal_rv.pdf(xy)
        z = z.reshape(self.num, self.num, order='F')
        z = z.T

         #
        #replace 'z'  gaussian by circle thresholded
        t = np.zeros((self.num,self.num))
        radius = 3
        for i in range(self.num):
            for j in range (self.num):
                if (X[i,j]**2 + Y[i,j]**2 <= radius**2):
                    t[i,j] = 1
                else:
                    t[i,j] = 0
                    
        #
        z = t
        Normed_z = [i/np.sum(z) for i in z]
        #extract the univariate Normal 
        uni_Normed_z = np.array(Normed_z)[:,int((self.num)/2)] #from the middle
        uni_Normed_z = uni_Normed_z/np.sum(uni_Normed_z) #again normalize

        fig, ax = plt.subplots(figsize=(5,5))
        plt.title('Thresholded blob')
        plt.contourf(X, Y, Normed_z,100)
        # plt.hist2d( z.T)
        plt.figure()
        plt.plot(np.linspace(0,len(uni_Normed_z)-1,len(uni_Normed_z)), uni_Normed_z)
        plt.show()

        self.static = Normed_z


        return X, Y, Normed_z, uni_Normed_z
    
    def partialY(self):
        #basically integrate Z distribution in X
        partial_inY = np.sum(np.array(self.static), 1)
        fig, ax = plt.subplots(figsize=(10,5))
        plt.title('Partial distribution in Y')
        plt.plot(np.linspace(0, len(partial_inY)-1,len(partial_inY)), partial_inY, 'r-')
       # plt.hist2d( z.T)
        plt.xlabel('pixels')
        plt.ylabel('Probability Density')
        plt.show()

        return partial_inY
    
    def do_moments(self,distribution, order):
        ''' order = moment(distribution,[0,1,2,3,4,5])
        print('The moments about this distribution are', order)
        '''
        c = 0
        moments = [] 
        central = 499.4999999
        for j,pw in enumerate(order):
            c=0
            for i,x in enumerate(distribution):
                 c = c + x* ((i - central) **pw)
            #c = c/100
            print('moment of order',j, ':', c)
            moments.append(c)

        return 



class streak(gaussian_blob):

    def __init__(self,amplitude,mean,spread,num,velocity,exposure,sampling):
        super().__init__(amplitude,mean,spread,num)
        self.exposure = exposure
        self.velocity = velocity
        self.sd = sampling
        self.num = num
    
    def convolve_gauss(self, Normed_z):

        self.static = Normed_z

        Grid = np.zeros((self.num*2,self.num*9)) #initialize grid
        pathy = int(Grid.shape[0]/2)
        path_length  = int(0.75 * Grid.shape[1])


        for i in range(int(path_length/self.sd)):  Grid[(pathy-(int(self.num/2))):(pathy +((int(self.num))-(int(self.num/2)))),( Grid.shape[1]-self.num-(int(self.num/2))-int((i)* self.sd)):(Grid.shape[1]-self.num-(int(self.num/2)) +int(self.num)-int((i)*self.sd))] += self.static

        #Normalize
        Grid = Grid#/np.sum(Grid)

        fig, ax = plt.subplots(2,figsize=(10,10))
        ax[0].imshow(Grid, interpolation='sinc', cmap='viridis')
        plt.title('Gaussian Convolution')

        Y_distribution = Grid[(int(self.num/2)):3*(int(self.num/2)),int(Grid.shape[1]/2)]#* path_length #need to account for the path length

        #fig, ax = plt.subplots(2,figsize=(15,10))
        ax[1].plot( np.linspace(0,len(Y_distribution)-1,len(Y_distribution)),Y_distribution)
        plt.title('Distribution in Y direction')
        plt.xlabel('pixels')
        plt.ylabel('Probability Density')

        return Grid, Y_distribution


# This is to calculate the partial distriution in Y for a set of experimentally measured Z-stack
class pointblur:

    def __init__(self, stack):
        self.stack = stack 

    def partialY(self, plot = False):
        #basically integrate in X
        collect = []
        for i,frame in enumerate(self.stack):
            partial_inY = np.sum(np.array(frame), 1)
            collect.append(partial_inY)

        if plot == True:
            fig, ax = plt.subplots(figsize=(10,5))
            plt.title('Partial distribution in Y')
            plt.plot(np.linspace(0, len(partial_inY)-1,len(partial_inY)), partial_inY, 'r-')
            # plt.hist2d( z.T)
            plt.xlabel('pixels')
            plt.ylabel('Probability Density')
            plt.show()

        self.partialYstack = collect   

    def fit_streak_height(self):
        #fit for the streak height
        collect = [] #reset
        for i in range(self.partialYstack.shape[0]):
            yy = self.partialYstack[i][:] #select all columns of ith row, aka the bump
            xx = np.arange(0,len(yy))

            #normalize 
            yy=(yy-np.min(yy))/(np.max(yy)-np.min(yy))

            #initial guess
            p0=[1,10,(len(yy)-20),0.01,0.01]

            #amp,w0,L,s,m,a,b
            try:
                pred_params, uncert_cov = curve_fit(gauss, xx, yy, p0=p0,method='lm')
                h = 1*pred_params[2]
                collect.append(h)

            except:
                h=0
                print('Rejected')
                collect.append(h)

        self.diameter = collect    






# functions

def do_blur(frame, blur = 5):
    
    blurred_image = gaussian(frame, blur)
    
    return blurred_image


def do_threshold(frame):
    
    blurred_img = do_blur(frame)
    value = skimage.filters.threshold_otsu(blurred_img)
    binary = frame > value
   # plt.imshow(binary, cmap="Greys_r", interpolation ="nearest") 

    return binary, value



def see_contour(frame, showplot = True):
    
    binary, value = do_threshold(frame)#otsu
    
    frame_ = do_blur(frame)
    contours = skimage.measure.find_contours(frame_, value)
    
    if showplot:
        
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap="Greys_r")
        
        for contour in contours:
            ax.plot(contour[:,1],contour[:,0], linewidth=1)
        
        
    return contours

# determine the properties using an n-sided discrete polygon
def propt_contour(x,y):
    
    
    Area = 0.5*( np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])  )
    fac = 1/(Area * 6)
    x_bar = fac *( np.sum((x[:-1]+x[1:])*(x[:-1]*y[1:] - x[1:]*y[:-1]))  )
    y_bar = fac*( np.sum((y[:-1]+y[1:])*(x[:-1]*y[1:] - x[1:]*y[:-1]))   )
    
    #print(x_bar, y_bar,Area)
    
    return Area, x_bar, y_bar


#fit a gaussian and filter the radius
def find_radius(frame, center):  #this is basically thresholding
    
    radial_dist = radial_profile(frame, center)
    
    #normalizing step
    
    radial_dist = (radial_dist - np.min(radial_dist))/np.max(radial_dist)
    x = np.arange(len(radial_dist))
    
    # take the radius at this threshold, intensity falls below this thresh is the critical limit
    #calculate mean of radial distribution
    mean = sum(radial_dist*x)/sum(radial_dist) #weighted sum by intensity count
    sigma = sum(radial_dist*(x-mean)**2)/(sum(radial_dist)) #standard deviation of distribution

    level = 0.2 #cutoff radius at 2 sigma

    
    try:
        interp_radial_dist = interpolate.interp1d(np.arange(len(radial_dist)), radial_dist)
    
        newx = np.linspace(0,len(radial_dist)-1,5000)
        resampled_radial_dist = interp_radial_dist(newx)
        radius = newx[np.min(np.where(resampled_radial_dist<level))]
        
    except:
        radius_pos = 0
    
    return radius


def intensity_inside_radius(frame, centroid, radius):
    
    mask = np.zeros(np.shape(frame))
    
    x = int(centroid[0]); y = int(centroid[1]);
    
    rr, cc = disk((x,y),radius, shape = np.shape(frame))
    
    mask[rr,cc]=1
    
    masked_img = mask*frame; #elementwise mulitplication
    
    intensity = np.sum(masked_img)
    
    return intensity   


def see_all_contours(frames, thresh = 100):
    
    radius = []; x_pos =[]; y_pos=[]; intensity =[]; intensity_mean=[]; framevec=[];
    
    for i, frame in enumerate(frames):
        
        if i%10 == 0:
            print('...processing image '+str(i)+' of '+str(len(frames)))
		
        contours = see_contour(frame, showplot=False)
        contour_size = []
        
        
        
        for contour in contours:
            x = contour[:,1] ; y = contour[:,0]
            Area, x_bar, y_bar = propt_contour(x,y)

            if Area > thresh: #element wise comparison filtering the area by some threshold value to elminate contours
                contour_size.append(Area)
                
            else:
                contour_size.append(0)
    
    
        if len(contour_size)>0:
            biggest = np.argmax(contour_size) #gives you the position of the biggest contour area
            x_big = contours[biggest][:,1] ; y_big = contours[biggest][:,0];
            
            Area_big, x_bar_big, y_bar_big = propt_contour(x_big, y_big);
            
            centroid_big = (x_bar_big, y_bar_big)
            
            r = find_radius(frame, centroid_big)
            
            radius.append(r) #capture this radius in a list
            
            x_pos.append(x_bar_big); y_pos.append(y_bar_big)
            
            ints = intensity_inside_radius(frame, centroid_big, r)
            
            intensity.append(ints)
            
            intensity_mean.append(ints/(np.pi*r**2))
                                  
            framevec.append(int(i))
            
            # #plot for clarity---
            # fig, ax = plt.subplots()
            # ax.imshow(frame, cmap="Greys_r")
            # ax.plot(x_big, y_big , linewidth=1)
            # ax.plot(x_bar_big, y_bar_big, color='r', linewidth='4')
            # print(centroid_big)
            # circle1 = plt.Circle((x_bar_big, y_bar_big),r,color='r', fill=False, linewidth=2)
            # ax.add_patch(circle1)
            
        else:
            
            radius.append(0) #capture this radius in a list            
            x_pos.append(0); y_pos.append(0)            
            intensity.append(0)            
            intensity_mean.append(0)                      
            framevec.append(int(i))
    
    
    all_contours = pd.DataFrame(data = np.vstack([x_pos, y_pos, radius, intensity, intensity_mean, framevec]).T, columns = ['x','y','radius','intensity','mean intensity','frame no'])
    return all_contours

#############################################################################################################################################################




def radial_profile(frame, center):
    
    y, x = np.indices(frame.shape)
    
    #this is basically the radius from center to all points
    r = np.sqrt((x- center[0])**2 + (y-center[1])**2)
    r = r.astype(int)# convert to integr

    weighted_bin = np.bincount(r.ravel(), frame.ravel())    
    nr = np.bincount(r.ravel())
    
    #Below is beasically a scaled version of Intensity sums along a contour
    # Radial profile = Intensity sum along a contour/ length of contour
    
    radialprof = weighted_bin/nr 
    
    return radialprof

 
    
    
    ################################################################################### fitting straight line
    
def calculate_slope(frame, radius):

    
      
    try:
        pred_params, uncert_cov = curve_fit(line, frame, radius,method='lm')
    except:
        pred_params = [0,0]

    	# #plt.figure()
    	# plt.plot(np.arange(len(profile)),profile/np.max(profile)+offset,marker='o',ms=2,ls='None',c=color,alpha=0.3)
    	# l = line(xx,*pred_params)
    	# plt.plot(xx, l/np.max(l)+offset, ls='--',c='k')

    return pred_params


def line(x,m,b):
    return m*x + b


def quadratic(x,a,m,b):
    return a*x*x + m*x + b
    
def hyperbolic(x,a,b):
    return (b + a * np.multiply(x,x))
   


# This function is for determining radius from a streak; it now only has a line profile 1D    
def radius_usingprofile(radial_dist, centerx):
    
    x = np.indices(radial_dist.shape)
    
    #this is basically the radius from center to all points
    r = np.sqrt((x- centerx)**2)
    r = r.astype(int)# convert to integr

    weighted_bin = np.bincount(r.ravel(), radial_dist.ravel())   

    nr = np.bincount(r.ravel())
    
    #Below is beasically a scaled version of Intensity sums along a contour
    # Radial profile = Intensity sum along a contour/ length of contour
    
    radialprof = weighted_bin/nr 
    

    #normalizing step
    
    radialprof = (radialprof - np.min(radialprof))
    radial_dist = radialprof/np.max(radialprof)

    # take the radius at this threshold, intensity falls below this thresh is the critical limit
    #level = 1.5*np.std(radial_dist) #currently 1.5 standard deviations
        #calculate mean of radial distribution
    x = np.arange(len(radial_dist))
    mean = sum(radial_dist*x)/sum(radial_dist) #weighted sum by intensity count
    sigma = np.sqrt(sum(radial_dist*(x-mean)**2)/(sum(radial_dist))) #standard deviation of distribution

    param, param_cov = curve_fit(gauss,x,radial_dist,p0=[max(radial_dist),mean,sigma])
    A = param[0]
    mean = param[1]
    sigma = param[2]
    radius = 2* sigma #cutoff radius at 2 sigma
    
    return radius,radial_dist, sigma, mean



# This function is for determining radius from a stack; it has 2D information  
def radius_usingprofile2D(frame, center):
    
    radial_dist = radial_profile(frame, center)
    
    #normalizing step
    
    radial_dist = (radial_dist - np.min(radial_dist))
    radial_dist = radial_dist/np.max(radial_dist)

        #calculate mean of radial distribution
    x = np.arange(len(radial_dist))
    mean = sum(radial_dist*x)/sum(radial_dist) #weighted sum by intensity count
    sigma = np.sqrt(sum(radial_dist*(x-mean)**2)/(sum(radial_dist))) #standard deviation of distribution

    param, param_cov = curve_fit(gauss,x,radial_dist,p0=[max(radial_dist),mean,sigma])
    A = param[0]
    mean = param[1]
    sigma = param[2]
    radius = 2* sigma #cutoff radius at 2 sigma
 
    
    return radius, radial_dist, sigma, mean




# a gaussian function
def gauss(x,A,mean,sigma):

    return A* np.exp(np.divide(-(np.square((x-mean))),(2*(sigma**2))))





#Morphology cv filtering

def morphimage(img,showplot):
    #stretch the dynamic range
    stretch = skimage.exposure.rescale_intensity(img, in_range='image',out_range=(0,255))
    

    #morph
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
    morph = cv2.morphologyEx(stretch,cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (200,200))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    #grayscale
    grey = morph.astype("uint8")

    #threshold
    thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    #largest contour
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bigcontour = max(contours, key = cv2.contourArea)

    #draw on black background
    contour = np.zeros((stretch.shape[:2]),dtype=np.uint8)
    result = np.zeros_like(stretch)
    cv2.drawContours(contour, [bigcontour], 0, 255,-1)
    result[contour>0] = stretch[contour>0]
    if showplot==True:
        plt.imshow(result)
      
    return result, thresh



