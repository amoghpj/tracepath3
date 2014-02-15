#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, linspace, zeros, array,\
                  concatenate, delete

from numpy.random import random, normal, randint, shuffle
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from scipy.spatial import cKDTree
import gtk, gobject

#np.random.seed(1)

SIZE = 20000
ONE = 1./SIZE

BACK = 1.

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

DIST_NEAR_INDICES = np.inf
NUM_NEAR_INDICES = 30
SHIFT_INDICES = 5

W = 0.9
PIX_BETWEEN = 11

START_X = (1.-W)*0.5
START_Y = (1.-W)*0.5

NUMMAX = int(2*SIZE)
NUM_LINES = int(SIZE*W/PIX_BETWEEN)
H = W/NUM_LINES

FILENAME = 'tt_brownianbridge'

TURTLE_ANGLE_NOISE = pi*0.1
INIT_TURTLE_ANGLE_NOISE = 0

def myrandom(size):

  #res = normal(size=size)

  #res = 1.-2.*random(size=size)

  ## almost but not entirely unlike a brownian bridge
  rnd = 1.-2.*random(size=size/2)
  res = concatenate((rnd,-rnd))
  shuffle(res)
  return res

def myrandom1():

  #res = normal(size=size)
  res = 1.-2.*random(size=1)

  return res


def turtle(sthe,sx,sy,steps):

  XY = zeros((steps,2),'float')
  THE = zeros(steps,'float')

  XY[0,0] = sx
  XY[0,1] = sy
  THE[0] = sthe
  the = sthe

  noise = myrandom(size=steps)*INIT_TURTLE_ANGLE_NOISE
  for k in xrange(1,steps):

    x = XY[k-1,0] + cos(the)*ONE
    y = XY[k-1,1] + sin(the)*ONE
    XY[k,0] = x
    XY[k,1] = y
    THE[k] = the
    the += noise[k]

    if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:
      XY = XY[:k,:]
      THE = THE[:k]
      break

  return THE, XY

#def get_near_indices(tree,xy,d,k):

  #dist,data_inds = tree.query(xy,k=k,distance_upper_bound=d,eps=ONE)

  #return dist, data_inds.flatten()

def get_near_indices(tree,xy,d,k):

  dist,data_inds = tree.query(xy,k=k,distance_upper_bound=d,eps=ONE)

  dist = dist.flatten()
  data_inds = data_inds.flatten()

  sort_inds = np.argsort(data_inds)

  dist = dist[sort_inds]
  data_inds = data_inds[sort_inds]

  return dist, data_inds.flatten()

def alignment(the,dist):

  dx = cos(the)
  dy = sin(the)

  ### inverse proporional distance scale
  #dx = np.sum(dx/dist)
  #dy = np.sum(dy/dist)

  ## linear distance scale
  dx = np.sum(dx*(1.-dist))
  dy = np.sum(dy*(1.-dist))

  dd = (dx*dx+dy*dy)**0.5

  return dx/dd,dy/dd


class Render(object):

  def __init__(self,n):

    self.n = n
    self.__init_cairo()

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

    self.ctx.set_source_rgba(0,0,0,0.6)
    self.ctx.set_line_width(ONE*3.)

    self.ctx.move_to(xy[0,0],xy[0,1])
    for (x,y) in xy[1:]:
      self.ctx.line_to(x,y)
    self.ctx.stroke()


def main():

  render = Render(SIZE)

  num = 0

  the = pi*0.5

  the,xy = turtle(the,START_X,START_Y,NUMMAX)

  render.line(xy)

  XYS = xy
  THES = the

  num = len(XYS)
  TS = cKDTree(XYS)

  for line_num in range(1,NUM_LINES):
    print line_num, NUM_LINES

    xy_res = zeros((NUMMAX,2),'float')
    the_res = zeros(NUMMAX,'float')
    
    #x = X_MIN + random()*(X_MAX-X_MIN)
    #y = Y_MIN + random()*(Y_MAX-Y_MIN)

    x = START_X+line_num*H
    y = START_Y

    xy_last = array([[x,y]])
    the_last = 0.5*pi

    xy_res[0,:] = xy_last
    the_res[0] = the_last

    noise = myrandom(size=NUMMAX)*TURTLE_ANGLE_NOISE

    for i in xrange(1,NUMMAX):

      dist,inds = get_near_indices(TS,xy_last,\
                                   DIST_NEAR_INDICES,\
                                   NUM_NEAR_INDICES)

      #dist[dist<ONE] = ONE
      dist = dist[SHIFT_INDICES:]
      inds = inds[SHIFT_INDICES:]

      dx,dy = alignment(THES[inds],dist)

      the = np.arctan2(dy,dx)
      the += noise[i]

      xy_new = xy_last + array( [[cos(the),sin(the)]] )*ONE
      xy_res[i,:] = xy_new
      the_res[i] = the

      xy_last = xy_new

      if xy_last[0,0]>X_MAX or xy_last[0,0]<X_MIN or\
         xy_last[0,1]>Y_MAX or xy_last[0,1]<Y_MIN:
        xy_res = xy_res[:i,:]
        the_res = the_res[:i]
        break

    render.line(xy_res)

    ## replace all old nodes
    XYS = xy_res
    THES = the_res

    num = len(THES)

    TS = cKDTree(XYS)

    if not line_num%100:

      render.sur.write_to_png('{:s}_{:d}.png'.format(FILENAME,line_num))

  render.sur.write_to_png('{:s}_final.png'.format(FILENAME))



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

