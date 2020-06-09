'''
Useful computer graphics functions.
'''

import numpy as np
import math


def rotation_matrix(axis, theta):
    '''
    Axis-angle rotation matrix.
    '''
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    return np.array([
        [
            a * a + b * b - c * c - d * d,
            2 * (b * c + a * d),
            2 * (b * d - a * c)
        ],
        [
            2 * (b * c - a * d),
            a * a + c * c - b * b - d * d,
            2 * (c * d + a * b),
        ],
        [
            2 * (b * d + a * c),
            2 * (c * d - a * b),
            a * a + d * d - b * b - c * c,
        ]
    ])


def rotate(v, axis, theta):
    '''
    Rotate `v` over `axis` an amount `theta` in radians.
    '''
    return np.dot(rotation_matrix(axis, theta), v)


def unit(v):
    '''
    Out-of-place unit vector conversion.
    '''
    result = np.array(v)
    normalize(result)
    return result


def normalize(v):
    '''In-place unit vector conversion.'''
    v[:] = v / np.linalg.norm(v, 2)
    return v


def angle_between(v1, v2):
    '''
    Robustly calculate angle between two vectors.

    '''
    # StackOverflow: 13849249/196284
    v1_u = unit(v1)
    v2_u = unit(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.
        else:
            return np.pi
    return angle


def uniform_sphere(rng, fov=math.pi):
    '''
    Sample uniformly over the unit sphere.

    Returns XYZ coordinates on the unit sphere's surface.

    '''
    s = rng.uniform(math.cos(fov), 1)
    phi = rng.uniform(0, 2 * math.pi)
    return np.array([s,
                     math.sqrt(1 - s ** 2) * math.cos(phi),
                     math.sqrt(1 - s ** 2) * math.sin(phi)])


def uniform_theta(rng, fov=math.pi):
    '''
    Sample uniformly over the angle between the sphere's primary axis, assumed
    to be [1, 0, 0].

    Returns XYZ coordinates on the unit sphere's surface.

    '''
    theta = rng.uniform(-fov, fov)
    phi = rng.uniform(0, 2 * math.pi)
    point = rotate([1., 0, 0], [0, 1, 0], theta)
    return rotate(point, [1, 0, 0], phi)


class Camera(object):
    '''
    A (mostly obsolete) Camera object. The plan was for this to define a
    coordinate system based on geometries, although, it wasn't clear how to do
    this. Instead, we define the coordinate system in the analysis scripts.
    Some old code still refers to this.
    '''

    def __init__(self, eye, view, up=(0, 1, 0), fov=60., aspect_ratio=1.):
        self.eye = eye
        self.view = view
        self.up = up
        self.fov = fov
        self.aspect_ratio = aspect_ratio

    @property
    def eye(self):
        return self._eye

    @eye.setter
    def eye(self, eye):
        self._eye = np.asarray(eye)

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, view):
        self._view = np.asarray(view)
        self._update()

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, up):
        self._up = np.asarray(up)
        self._update()

    @property
    def right(self):
        return self._right

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        self._fov = fov
        self._update_fov()

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, aspect_ratio):
        self._aspect_ratio = aspect_ratio
        self._update_fov()

    def cast(self, imx, imy):
        """
        Cast a ray through the image plane

        Parameters:
          imx: normalized x image component
          imy: normalized y image component
        """
        if not 0 <= imx <= 1 or not 0 <= imy <= 1:
            raise ValueError('imx and imy must be in range [0, 1]')
        x = (2. * imx - 1.) * self._tan_hfov
        y = (2. * imy - 1.) * self._tan_vfov
        return normalize(y * self.up + x * self.right + self.view)

    def _update(self):
        try:
            up = self.up
            view = self.view
        except AttributeError:
            return
        normalize(view)
        up = unit(up - up.dot(view) * view)
        right = unit(np.cross(view, up))
        self._up = up
        self._view = view
        self._right = right

    def _update_fov(self):
        try:
            fov = self.fov
            aspect_ratio = self.aspect_ratio
        except AttributeError:
            return
        self._tan_hfov = math.tan(fov * math.pi / 180.0 / 2)
        self._tan_vfov = self._tan_hfov / aspect_ratio
