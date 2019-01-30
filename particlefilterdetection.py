# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 01:28:19 2018

@author: sigul
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from itertools import izip
import time
import pylab 
from numpy.random import *

def addnoise(image, strength):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.01 * strength
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
    for i in image.shape]
    out[tuple(coords)] = 1
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
    for i in image.shape]
    out[tuple(coords)] = 0
    return out

# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html    
def resample(weights):
  n = len(weights)
  indices = []
  C = [0.] + [sum(weights[:i+1]) for i in range(n)]
  u0, j = random(), 0
  for u in [(u0+i)/n for i in range(n)]:
    while u > C[j]:
      j+=1
    indices.append(j-1)
  return indices

def particlefilter(sequence, pos, stepsize, n):
  seq = iter(sequence)
  x = ones((n, 2), int) * pos                   # Initial position
  f0 = seq.next()[tuple(pos)] * ones(n)         # Target colour model
  yield pos, x, ones(n)/n                       # Return expected position, particles and weights
  for im in seq:
    np.add(x, uniform(-stepsize, stepsize, x.shape), out=x, casting="unsafe")  # Particle motion model: uniform step
    x  = x.clip(zeros(2), array(im.shape)-1).astype(int) # Clip out-of-bounds particles
    f  = im[tuple(x.T)]                         # Measure particle colours
    w  = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
    w /= sum(w)                                 # Normalize w
    yield sum(x.T*w, axis=1), x, w              # Return expected position, particles and weights
    if 1./sum(w**2) < n/2.:                     # If particle cloud degenerate:
      x  = x[resample(w),:]                     # Resample particles according to weights


clean_image = mpimg.imread("room.jpg")          # Read the original image
target = mpimg.imread("switch.jpg");            # Read the part we're looking for

images = [];

for i in range(1, 20):                          # Fill the array with a sequence of 20 gradually more noisy images
    noisy_image = addnoise(clean_image, i)
    images[i] = noisy_image
    
seq = iter(images)
x = ones((n, 2), int) * pos                     # Initial position
f0 = seq.next()[tuple(pos)] * ones(n)