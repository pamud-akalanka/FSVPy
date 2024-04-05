import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import pandas as pd
import numpy as np
import matplotlib.patches as patches


###################################################################################
'''
plot_contours - to be used to plot contours, for centerlines see plot_centerlines

inputs:
    im: image of streaks, numpy array
    contours: list of contours, each list entry is numpy array of x,y positions in contour

output: None, displays image
'''
###################################################################################

def plot_streaks(contours, im):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap = 'Greys_r')

    for i, contour in enumerate(contours):
        ax.plot(contour[:,1], contour[:,0], linewidth=2)
        ax.text(contour[0][1], contour[0][0], str(int(i)), fontsize = 12, c = 'y')
    plt.show()

###################################################################################
'''
plot_contours_label - to be used to plot contours that have been characterized by running parameters

inputs:
    df: pandas dataframe with streak information (output of measure.parameters)
    im: image of streaks, numpy array
    contours: list of contours, each list entry is numpy array of x,y positions in contour

output: None, displays image
'''
###################################################################################

def plot_filtered_streaks(contours, df, im):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap = 'Greys_r')

    for ind in df.streak_id.unique():

        contour = contours[int(ind)]
        ax.plot(contour[:,1], contour[:,0], linewidth=1 )
        ax.text(contour[0][1], contour[0][0], str(int(ind)), fontsize = 12, c = 'y')
        
    plt.show()

###################################################################################
'''
plot_fit - plots width and length identified for straight streaks

inputs:
    df: pandas dataframe with streak information
    im: image of streaks, numpy array
    contours: list of contours, each list entry is numpy array of x,y positions in contour

output: None, displays image
'''
###################################################################################

def plot_fit(im, df):

    fig, ax = plt.subplots()
    ax.imshow(im, cmap = 'Greys_r')
    
    for idx, row in df.iterrows():
        cx = row.x; cy = row.y;  #centroid
        theta = -row.angle*np.pi/180 #rotation angle
        #plot width line
        lx, ly = rotate(cx - row.width/2, cy, cx, cy, theta)
        rx, ry = rotate(cx + row.width/2, cy, cx, cy, theta)
        plt.plot([lx,rx],[ly,ry],c='b',linewidth=0.5)
        #plot height line
        lx, ly = rotate(cx, cy - row.height/2, cx, cy, theta)
        rx, ry = rotate(cx, cy + row.height/2, cx, cy, theta)
        plt.plot([lx,rx],[ly,ry],c='b',linewidth=0.5)
        ax.text(row.x, row.y, str(int(row.streak_id)), fontsize = 12,c='y')
        #plot bbox using detected contour
        rect = patches.Rectangle((row.corner_x,row.corner_y-row.bbox_height/2), row.bbox_width, row.bbox_height,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        #plot bbox with padding (I think is used for fitting purposes)



    plt.show()


#helper function for plot_fit
def rotate(x,y,cx,cy,theta):  #theta should be in radians
    xp = ((x-cx)*np.cos(theta) + (y - cy)*np.sin(theta)) + cx
    yp = (-(x-cx)*np.sin(theta) + (y - cy)*np.cos(theta)) + cy

    return xp, yp



###################################################################################
#plot_centerline - plots the centerline found for curvy streaks

'''
inputs:
    cl: list of centerlines, list of numpy arrays
    im: image of streaks, numpy array
    show_point_order: plot centerline points using a colormap to indicate the ordering of points, dafualt is True
    cmap: colormap for centerline to indicate point order, default is rainbow
    color: color for streaks if you do not want to indicate point order, default is red

output: None, displays image
'''
###################################################################################


def plot_centerlines(cl, image, show_point_order = False, color = 'r', cmap = 'rainbow'):
    fig, ax = plt.subplots()
    plt.imshow(image, cmap = 'Greys_r')

    for i, item in enumerate(cl):
        if len(item) > 0:
            if show_point_order:
                ax.scatter(item[:,1],item[:,0],s=5,marker='o',c=np.arange(len(item)), cmap = cmap)  
                ax.text(item[0][1], item[0][0], str(int(i)), fontsize = 12, c = 'y')
            else:
                ax.plot(item[:,1],item[:,0],c=color)
                ax.text(item[0][1], item[0][0], str(int(i)), fontsize = 12, c = 'y')
    plt.show()


#Good for debugging
####################

def plot_contours(image,contours):

    fig, ax = plt.subplots()
    ax.imshow(image, cmap = 'Greys_r')

    for contour in contours:
        ax.plot(contour[:,1], contour[:,0], linewidth=0.5)

    plt.show()    



