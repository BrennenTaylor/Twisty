#!/usr/bin/python

import math

def f3_formula(r):
    lower = abs(r-1.0)
    return r+1.0 - lower;



def f4_integral(r):
    lower = abs(r-1.0)
    upper = r+1.0
    nb_points = 1000
    dr = (upper-lower)/(nb_points-1)
    accum = 0.0
    for i in range(0,nb_points):
        rp = dr*i
        accum = accum + dr*rp*f3_formula(rp)
    return accum



def fn_integral(r, order):
    lower = abs(r-1.0)
    upper = r+1.0
    nb_points = 1000
    dr = (upper-lower)/(nb_points-1)
    accum = 0.0
    if order==4:
        for i in range(0,nb_points):
            rp = dr*i
            accum = accum + dr*rp*f3_formula(rp)
    else:
        for i in range(0,nb_points):
            rp = dr*i
            accum = accum + dr*rp*fn_integral(rp,order-1)
    return accum



nb_points = 1000
dr = 1.0/(nb_points-1)
for i in range(0,nb_points):
    r = i*dr
    f3 = f3_formula(r)
    print str(r) + " " + str(f3) 

print "\n\n\n\n"
for i in range(0,nb_points):
    r = i*dr
    f4 = f4_integral(r)
    print str(r) + " " + str(f4)


print "\n\n\n\n"
for i in range(0,nb_points):
    r = i*dr
    f5 = fn_integral(r,5)
    print str(r) + " " + str(f5)


print "\n\n\n\n"
for i in range(0,nb_points):
    r = i*dr
    f6 = fn_integral(r,6)
    print str(r) + " " + str(f6)

