import config
from gfx import uniform_sphere, unit, uniform_theta

import numpy as np

from abc import ABCMeta, abstractmethod


class Geometry(config.Element):
    '''
    A distribution of points and directions. Meant for modelling sensors and
    emitters.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def choose(self, rng):
        '''
        Pick a random point and direction defined by the geometry.

        Parameters:
          rng (Random): a random number generator

        Returns:
          point: numpy.ndarray a point in 3-space on the geometry
          direction: numpy.ndarray a direction in 3-space on the geometry

        '''
        raise NotImplementedError


class Ray(Geometry):
    '''
    Always returns `origin` and `direction` for `choose()`.
    '''

    def __init__(self, origin, direction):
        self.origin = np.asarray(origin)
        self.direction = np.asarray(direction)

    def choose(self, _):
        return np.array(self.origin), np.array(self.direction)

    def to_dict(self):
        return {
            'type': self.get_type(),
            'origin': self.origin.tolist(),
            'direction': self.direction.tolist(),
        }


class Sphere(Geometry):
    '''
    Returns points on the sphere and surface normal at that point. Supports
    spherical uniform sampling and "theta-uniform" sampling.

    Parameters
    ----------

    radius: float
        the radius of the sphere
    center: array of size 3
        the center of the sphere
    basis: 3x3 matrix or none
        a 3d rotation matrix for the sphere; by the default the sphere's pole
        is in the x direction
    fov: float
        angle from the pole to support; by default the entire sphere (pi)
    uniform_in: str
        sampling strategy; either 'sphere' or 'theta'
    '''

    def __init__(self, radius, center=(0, 0, 0), basis=None, fov=np.pi,
                 uniform_in='sphere'):
        self.center = np.asarray(center)
        self.radius = radius
        if basis is None:
            basis = np.eye(3)
        self.basis = np.asarray(basis)
        self.fov = fov
        self.uniform_in = uniform_in

    def choose(self, rng):
        if self.uniform_in == 'sphere':
            xform = np.dot(self.basis, uniform_sphere(rng, self.fov))
        elif self.uniform_in == 'theta':
            xform = np.dot(self.basis, uniform_theta(rng, self.fov))
        else:
            raise ValueError('Invalid uniform_in value: %s' % self.uniform_in)
        point = self.center + self.radius * xform
        direction = unit(point - self.center)
        return point, direction

    def to_dict(self):
        return {
            'type': self.get_type(),
            'center': self.center.tolist(),
            'direction': self.direction.tolist(),
            'fov': self.fov,
            'basis': self.basis.tolist(),
            'uniform_in': self.uniform_in,
        }
