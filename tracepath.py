#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, linspace, zeros, array,\
                  concatenate, delete

from numpy.random import random, normal, randint
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from scipy.spatial import cKDTree
import gtk, gobject

np.random.seed(1)

SIZE = 1000
ONE = 1./SIZE

BACK = 1.

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

TURTLE_ANGLE_NOISE = pi*0.03
INIT_TURTLE_ANGLE_NOISE = pi*0.003

DIST_NEAR_INDICES = np.inf
NUM_NEAR_INDICES = 40

W = 0.9
PIX_BETWEEN = 6.

START_X = (1.-W)*0.5
START_Y = (1.-W)*0.5

NUMMAX = 2*SIZE
NUM_LINES = int(SIZE*W/PIX_BETWEEN)
H = W/NUM_LINES


def turtle(sthe,sx,sy,steps):

  XY = zeros((steps,2),'float')
  THE = zeros(steps,'float')

  XY[0,0] = sx
  XY[0,1] = sy
  THE[0] = sthe
  the = sthe

  for k in xrange(1,steps):

    x = XY[k-1,0] + cos(the)*ONE
    y = XY[k-1,1] + sin(the)*ONE
    XY[k,0] = x
    XY[k,1] = y
    THE[k] = the
    the += normal()*INIT_TURTLE_ANGLE_NOISE

    if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:
      XY = XY[:k,:]
      THE = THE[:k]
      break

  return THE, XY

def get_near_indices(tree,xy,d,k):

  dist,data_inds = tree.query(xy,k=k,distance_upper_bound=d,eps=ONE)

  return dist, data_inds.flatten()


class Render(object):

  def __init__(self,n):

    self.n = n
    self.itt = 0

    self.__init_cairo()
    self.__init_data()

    self.num_img = 0

  def __init_data(self):

    self.num = 0

    the = pi*0.5

    the,xy = turtle(the,START_X,START_Y,NUMMAX)

    self.line(xy)

    self.XYS = xy
    self.THES = the

    self.num = len(self.XYS)
    self.TS = cKDTree(self.XYS)

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def line(self,xy):

    self.ctx.set_source_rgba(0,0,0,0.8)
    self.ctx.set_line_width(ONE)

    self.ctx.move_to(xy[0,0],xy[0,1])
    for (x,y) in xy[1:]:
      self.ctx.line_to(x,y)
    self.ctx.stroke()

  def alignment(self,the,dist):

    dx = cos(the)
    dy = sin(the)
    ## inverse proporional distance scale
    dx = np.sum(dx/dist)
    dy = np.sum(dy/dist)

    dd = (dx*dx+dy*dy)**0.5

    return dx/dd,dy/dd

  def step(self):

    self.itt += 1

    xy_res = zeros((NUMMAX,2),'float')
    the_res = zeros(NUMMAX,'float')
    
    #x = X_MIN + random()*(X_MAX-X_MIN)
    #y = Y_MIN + random()*(Y_MAX-Y_MIN)

    x = START_X+self.itt*H
    y = START_Y

    xy_last = array([[x,y]])
    the_last = random()*pi*2.

    xy_res[0,:] = xy_last
    the_res[0] = the_last

    #noise = normal(size=NUMMAX)*TURTLE_ANGLE_NOISE

    ## inverse the travel direction for every 2nd line.
    ## but not the angle in the unstructured vector field (A)
    #switch_dir = pi if not random()<0.5 else 0
    switch_dir = 0

    for i in xrange(1,NUMMAX):

      k = NUM_NEAR_INDICES
      d =  DIST_NEAR_INDICES
      dist,inds = get_near_indices(self.TS,xy_last,d,k)

      dx,dy = self.alignment(self.THES[inds],dist)

      the = np.arctan2(dy,dx)
      #the += noise[i]

      xy_new = xy_last + array( [[cos(the+switch_dir),\
                                  sin(the+switch_dir)]] )*ONE
      xy_res[i,:] = xy_new
      the_res[i] = the

      xy_last = xy_new

      if xy_last[0,0]>X_MAX or xy_last[0,0]<X_MIN or\
         xy_last[0,1]>Y_MAX or xy_last[0,1]<Y_MIN:
        xy_res = xy_res[:i,:]
        the_res = the_res[:i]
        break

    self.line(xy_res)

    ## replace all old nodes
    self.XYS = xy_res
    self.THES = the_res

    self.num = len(self.THES)

    self.TS = cKDTree(self.XYS)

def main():

  render = Render(SIZE)
  for i in range(NUM_LINES):
    print i, NUM_LINES
    render.step()

  render.sur.write_to_png('res.png')



if __name__ == '__main__':
  if False:
    import pstats
    import cProfile
    OUT = 'profile'
    pfilename = '{:s}.profile'.format(OUT)
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()

