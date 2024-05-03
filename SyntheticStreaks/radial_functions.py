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
def find_radius(frame, center):
    
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
    
def hyperbolic(x,a,b):
    return (b + a * np.multiply(x,x))
   
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


def fit_gauss(x, y, mean, sigma):
    

    return