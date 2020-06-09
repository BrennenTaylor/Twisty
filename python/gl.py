"""
Convenient OpenGL drawing routines.

Primarily used by twisty-show.py
"""
import OpenGL.GL as gl
import OpenGL.GLU as glu
import math
import numpy as np

def draw_curve(curve):
    """ Draw a curve or otherwise sequence of points.
    """
    if isinstance(curve, FS):
        draw_curve(curve.points())
    gl.glBegin(gl.GL_LINE_STRIP)
    for p in curve:
        gl.glVertex3f(*p)
    gl.glEnd()


def draw_grid(m, n):
    """ Draw a m by n grid on the y=0 plane.
    """
    x0 = int(-m / 2)
    z0 = int(-n / 2)
    gl.glBegin(gl.GL_QUADS)
    for x in range(m):
        for z in range(n):
            gl.glVertex3f(x0 + x, 0, z0 + z)
            gl.glVertex3f(x0 + x + 1, 0, z0 + z)
            gl.glVertex3f(x0 + x + 1, 0, z0 + z + 1)
            gl.glVertex3f(x0 + x, 0, z0 + z + 1)
    gl.glEnd()