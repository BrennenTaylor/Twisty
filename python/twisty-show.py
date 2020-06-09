# Modify the viewer to view a single path

#!/usr/bin/env python
from __future__ import division, print_function
import sys
from OpenGL.GL import *  # noqa
from OpenGL.GLU import *  # noqa
from OpenGL.GLUT import *  # noqa
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv

import math
import pprint
from collections import defaultdict
from argparse import ArgumentParser

from util import attrdict, timeit
from gl import draw_curve, draw_grid
import geometry

# histogram display constants
NONE, WEIGHTS, PARAMETERS, BOTH = HISTOGRAM_MODES = range(4)

# Log level parameters
DEBUG, INFO, WARN, ERROR = 10, 20, 30, 40

# XXX These constants are not standard
WHEEL_UP, WHEEL_DOWN = 3, 4

XYZ_COLORS = [
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
]

SPHERE_QUADRIC = gluNewQuadric()

USAGE = '''\
Mouse commands:

    Left mouse button: Tumble
    Scroll wheel:      Zoom

Keyboard commands:

    f: Output frames of the OpenGL buffer.
    g: Toggle grid.
    p: Print metadata.
    r: Toggle rotation minimizing frames.
    s: Toggle geometry display (if available).
    t: Toggle turntable.
    x: Toggle origin display.
    z: Reset the view.
    2: Toggle seed path display.
    ?: Print this message.
'''

runner = None
opt = attrdict()
opt.current_frame = 1
opt.frame_format = "out.%04d.png"
opt.history = []
opt.history_max = 256
opt.meta = attrdict()
opt.refresh_rate = int(1000 / 30)
opt.rotate_x = 0.0
opt.rotate_y = 0.2
opt.turntable_rate = math.pi / 10
opt.want_continuous = False
opt.want_frames = False
opt.want_geometry = True
opt.want_grid = True
opt.want_xyz = False
opt.want_history = False
opt.want_rmf = False
opt.want_turntable = False
opt.zoom = 1.
opt.log_level = INFO

ui = attrdict()
ui.rotating = False

WARNING_FORMAT = """
{sep}
Something seems to be the matter!
{sep}

The solution {did_or_did_not} converge, yet...

- The absolute difference in the end point is {err1}
- The absolute difference in the end tangent is {err2}

Here is the result from scipy:

{result}
"""


def log(level, *args, **kw):
    if level >= opt.log_level:
        print(*args, **kw)


def debug(*args, **kw):
    log(DEBUG, *args, **kw)


def info(*args, **kw):
    log(INFO, *args, **kw)


def warn(*args, **kw):
    log(WARN, *args, **kw)

class Segment:

    def __init__(self):
        self.Pos = np.zeros(3)
        self.Frame = np.zeros((3, 3))
        self.Rotation = np.identity(3)
        self.Length = 0.0

def setup(args):
    # We want to store the curve information here
    with open(args.curve, 'r') as f:
        curveReader = csv.reader(f, delimiter=',')
        curveRows = list(curveReader)

    with open(args.bezier, 'r') as bc_csv:
        bezierReader = csv.reader(bc_csv, delimiter=',')
        bezierRows = list(bezierReader)

    opt.meta.beziercp0 = np.array([float(bezierRows[1][0]), float(bezierRows[1][1]), float(bezierRows[1][2])])
    opt.meta.beziercp1 = np.array([float(bezierRows[2][0]), float(bezierRows[2][1]), float(bezierRows[2][2])])
    opt.meta.beziercp2 = np.array([float(bezierRows[3][0]), float(bezierRows[3][1]), float(bezierRows[3][2])])
    opt.meta.beziercp3 = np.array([float(bezierRows[4][0]), float(bezierRows[4][1]), float(bezierRows[4][2])])
    opt.meta.beziercp4 = np.array([float(bezierRows[5][0]), float(bezierRows[5][1]), float(bezierRows[5][2])])

    opt.meta.numSegments = int(curveRows[1][1])
    opt.meta.arclength = float(curveRows[2][1])
    opt.meta.basePos = np.array([float(curveRows[4][1]), float(curveRows[4][2]), float(curveRows[4][3])])

    opt.meta.baseFrame = np.array\
    ([\
        [float(curveRows[5][1]), float(curveRows[5][2]), float(curveRows[5][3])],\
        [float(curveRows[6][1]), float(curveRows[6][2]), float(curveRows[6][3])],\
        [float(curveRows[7][1]), float(curveRows[7][2]), float(curveRows[7][3])],\
    ])

    #TODO:  Bad design
    segmentRowStart = 10
    numRowsPerSegment = 12

    opt.meta.segments = []
    for i in range(0, opt.meta.numSegments):
        opt.meta.segments.append(Segment())

    print ("Num Segments")
    print(len(opt.meta.segments))

    # Loop and load in the segment data
    for i in range(0, opt.meta.numSegments):
        currentSegmentStartRow = segmentRowStart + i * numRowsPerSegment

        opt.meta.segments[i].Length = float(curveRows[currentSegmentStartRow+1][1])
        opt.meta.segments[i].curvature = float(curveRows[currentSegmentStartRow+2][1])
        opt.meta.segments[i].torsion = float(curveRows[currentSegmentStartRow+3][1])

        # opt.meta.segments[i].position = np.array([float(curveRows[currentSegmentStartRow+4][1]),\
        #     float(curveRows[currentSegmentStartRow+4][2]),\
        #     float(curveRows[currentSegmentStartRow+4][3])])

        opt.meta.segments[i].tangent = np.array([float(curveRows[currentSegmentStartRow+4][1]),\
            float(curveRows[currentSegmentStartRow+4][2]),\
            float(curveRows[currentSegmentStartRow+4][3])])

        opt.meta.segments[i].normal = np.array([float(curveRows[currentSegmentStartRow+5][1]),\
            float(curveRows[currentSegmentStartRow+5][2]),\
            float(curveRows[currentSegmentStartRow+5][3])])

        opt.meta.segments[i].binormal = np.array([float(curveRows[currentSegmentStartRow+6][1]),\
            float(curveRows[currentSegmentStartRow+6][2]),\
            float(curveRows[currentSegmentStartRow+6][3])])


        opt.meta.segments[i].rotationMatrix = np.array\
            ([\
                [float(curveRows[currentSegmentStartRow+7][1]), float(curveRows[currentSegmentStartRow+7][2]), float(curveRows[currentSegmentStartRow+7][3])],\
                [float(curveRows[currentSegmentStartRow+8][1]), float(curveRows[currentSegmentStartRow+8][2]), float(curveRows[currentSegmentStartRow+8][3])],\
                [float(curveRows[currentSegmentStartRow+9][1]), float(curveRows[currentSegmentStartRow+9][2]), float(curveRows[currentSegmentStartRow+9][3])]\
            ])

    # Draw the curve
    currentFrame = np.copy(opt.meta.baseFrame)
    currentPos = np.copy(opt.meta.basePos)

    for i in range(0, opt.meta.numSegments):
        print(currentPos)
        currentPos = currentPos + opt.meta.segments[i].Length * currentFrame[0, :] #opt.meta.segments[i].tangent
        print(currentPos)

        print(currentFrame)
        currentFrame = np.matmul(opt.meta.segments[i].rotationMatrix, currentFrame)
        print(currentFrame)


        # Note: See if calculated frame and position is correct
        opt.meta.segments[i].position = currentPos
        opt.meta.segments[i].tangent = currentFrame[0, :]
        opt.meta.segments[i].normal = currentFrame[1, :]
        opt.meta.segments[i].binormal = currentFrame[2, :]

    # print("Start Pos")
    # print(opt.meta.basePos)
    # print("Final Pos")
    # print(currentPos)


def dispatch_write_frame():
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, opt.width, opt.height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.fromstring("RGBA", (opt.width, opt.height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    write_dict(image, opt.meta)
    image.save(opt.frame_format % opt.current_frame,
               opt.frame_format.rsplit(".", 1)[-1].upper())
    debug("Wrote %s" % (opt.frame_format % opt.current_frame))
    opt.current_frame += 1

def init():
    pass


def reset_projection():
    """ Place camera a clip pane length away, with a tilt of about theta.
    """
    width, height = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0 * opt.zoom, width / height, 0.1, 100.0)
    l = 100.0 / 2
    theta = 30.0 * math.pi / 180.0
    lz, ly = l * math.cos(theta), l * math.sin(theta)
    gluLookAt(0, ly, lz,
              0, 0, 0,
              0, 1, 0)


def keyboard(key, x, y):
    def _toggle(label, state):
        info('%s: %s' % (label, 'ON' if state else 'OFF'))
    # if key.lower() == 'f':
    #     opt.want_frames = not opt.want_frames
    #     _toggle('Frame output', opt.want_frames)
    if key.lower() == 'g':
        opt.want_grid = not opt.want_grid
        _toggle('Grid', opt.want_grid)
        glutPostRedisplay()
    if key.lower() == 'p':
        pprint.pprint(opt.meta)
    if key.lower() == 'r':
        opt.want_rmf = not opt.want_rmf
        _toggle('Rotation minimizing frame', opt.want_rmf)
    if key.lower() == 's':
        opt.want_geometry = not opt.want_geometry
        _toggle('Geometry', opt.want_geometry)
        glutPostRedisplay()
    if key.lower() == 't':
        opt.want_turntable = not opt.want_turntable
        _toggle('Turntable', opt.want_turntable)
    if key.lower() == 'x':
        opt.want_xyz = not opt.want_xyz
        _toggle('Origin XYZ', opt.want_xyz)
        glutPostRedisplay()
    if key.lower() == 'z':
        info('Resetting view')
        reset_view()
        glutPostRedisplay()
    if ord(key) == 27:
        sys.exit(0)
    if key.lower() == '?':
        print('=' * 80)
        print(USAGE)
        print('=' * 80)


def timer(value):
    if opt.want_turntable:
        opt.rotate_y += opt.turntable_rate
        glutPostRedisplay()
    if opt.want_continuous:
        dispatch_continuous()
        glutPostRedisplay()
    glutTimerFunc(opt.refresh_rate, timer, 0)


def mouse(button, state, x, y):
    if button == GLUT_LEFT_BUTTON:
        ui.rotating = state == GLUT_DOWN
    elif button == WHEEL_UP:
        opt.zoom += 0.01
        glutPostRedisplay()
    elif button == WHEEL_DOWN:
        opt.zoom = max(0.01, opt.zoom - 0.01)
        glutPostRedisplay()
    if ui.rotating:
        ui.original_rotate = opt.rotate_x, opt.rotate_y
        ui.rotate_start = x, y


def motion(x, y):
    if ui.rotating:
        opt.rotate_x = ui.original_rotate[0] + 0.1 * (ui.rotate_start[1] - y)
        opt.rotate_y = ui.original_rotate[1] + 0.1 * (ui.rotate_start[0] - x)
        glutPostRedisplay()


def display():
    reset_projection()
    glMatrixMode(GL_MODELVIEW)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glRotate(opt.rotate_x, 1, 0, 0)
    glRotate(opt.rotate_y, 0, 1, 0)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glColor(1, 1, 1, 1)
    if opt.want_grid:
        draw_grid(10, 10)
    if opt.want_xyz:
        draw_xyz()

    if opt.want_geometry:
        draw_scene()

    # write frames
    if opt.want_frames:
        dispatch_write_frame()

    glutSwapBuffers()


def draw_scene():


    # Draw the base curve basis
    glBegin(GL_LINES)
    scale = 0.5
    glColor(1, 0, 0, 1)
    glVertex3f(opt.meta.basePos[0], opt.meta.basePos[1], opt.meta.basePos[2])
    tangent = opt.meta.basePos + opt.meta.baseFrame[0,:] * scale
    glVertex3f(tangent[0], tangent[1], tangent[2])

    glColor(0, 1, 0, 1)
    glVertex3f(opt.meta.basePos[0], opt.meta.basePos[1], opt.meta.basePos[2])
    normal = opt.meta.basePos + opt.meta.baseFrame[1,:] * scale
    glVertex3f(normal[0], normal[1], normal[2])

    glColor(0, 0, 1, 1)
    glVertex3f(opt.meta.basePos[0], opt.meta.basePos[1], opt.meta.basePos[2])
    binormal = opt.meta.basePos + opt.meta.baseFrame[2,:] * scale
    glVertex3f(binormal[0], binormal[1], binormal[2])

    # Now, we want to loop through all the segments and build that up as well
    for i in range(0, opt.meta.numSegments):
        segment = opt.meta.segments[i]
        scale = segment.Length

        glColor(1, 0, 0, 1)
        glVertex3f(segment.position[0], segment.position[1], segment.position[2])
        tangent = segment.position + segment.tangent * scale
        glVertex3f(tangent[0], tangent[1], tangent[2])

        glColor(0, 1, 0, 1)
        glVertex3f(segment.position[0], segment.position[1], segment.position[2])
        normal = segment.position + segment.normal * scale
        glVertex3f(normal[0], normal[1], normal[2])

        glColor(0, 0, 1, 1)
        glVertex3f(segment.position[0], segment.position[1], segment.position[2])
        binormal = segment.position + segment.binormal * scale
        glVertex3f(binormal[0], binormal[1], binormal[2])
    glEnd()


    # Now draw control pts
    cpRadius = 0.3

    glPushMatrix()
    glTranslate(opt.meta.beziercp0[0], opt.meta.beziercp0[1], opt.meta.beziercp0[2])
    glColor(1.0, 1.0, 1.0, 1)
    gluSphere(SPHERE_QUADRIC, cpRadius, 20, 20)
    glPopMatrix()

    glPushMatrix()
    glTranslate(opt.meta.beziercp1[0], opt.meta.beziercp1[1], opt.meta.beziercp1[2])
    glColor(1.0, 0.0, 0.0, 1)
    gluSphere(SPHERE_QUADRIC, cpRadius, 20, 20)
    glPopMatrix()

    glPushMatrix()
    glTranslate(opt.meta.beziercp2[0], opt.meta.beziercp2[1], opt.meta.beziercp2[2])
    glColor(0.0, 1.0, 0.0, 1)
    gluSphere(SPHERE_QUADRIC, cpRadius, 20, 20)
    glPopMatrix()

    glPushMatrix()
    glTranslate(opt.meta.beziercp3[0], opt.meta.beziercp3[1], opt.meta.beziercp3[2])
    glColor(0.0, 0.0, 1.0, 1)
    gluSphere(SPHERE_QUADRIC, cpRadius, 20, 20)
    glPopMatrix()

    glPushMatrix()
    glTranslate(opt.meta.beziercp4[0], opt.meta.beziercp4[1], opt.meta.beziercp4[2])
    glColor(1.0, 1.0, 1.0, 1)
    gluSphere(SPHERE_QUADRIC, cpRadius, 20, 20)
    glPopMatrix()

def draw_basis(x, basis, scale, colors=None):
    colors = colors or [None] * 4
    glBegin(GL_LINES)
    for c, v in zip(colors, basis):
        if c:
            glColor(*c)
        glVertex3f(*x)
        glVertex3f(*(x + scale * v))
    glEnd()


def draw_xyz(scale=0.5):
    draw_basis(np.zeros(3), np.eye(3), scale, colors=XYZ_COLORS)


def reset_view():
    opt.rotate_x = 0.0
    opt.rotate_y = 0.2
    opt.zoom = 0.3


def _hsv_to_rgb(hsv):
    h, s, v = hsv
    hp = h / 60.0
    c = v * s
    x = c * (1 - abs(hp % 2 - 1))
    if 0 <= hp < 1:
        return c, x, 0
    elif 1 <= hp < 2:
        return x, c, 0
    elif 2 <= hp < 3:
        return 0, c, x
    elif 3 <= hp < 4:
        return 0, x, c
    elif 4 <= hp < 5:
        return x, 0, c
    elif 5 <= hp < 6:
        return c, 0, x
    else:
        return 0, 0, 0


def main():
    glutInit(sys.argv)
    parser = ArgumentParser()
    parser.add_argument('--curve', help='curve file to use')
    parser.add_argument('--bezier', help='bezier file to use')
    parser.add_argument('--dimensions', nargs=2, type=int, default=(960, 540))
    parser.add_argument('--fullscreen', action='store_true')
    args = parser.parse_args()
    setup(args)
    opt.width, opt.height = args.dimensions

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    if args.fullscreen:
        glutEnterGameMode()
    else:
        glutInitWindowSize(*args.dimensions)
        glutCreateWindow('Curve visualizer')

    glClearColor(0., 0., 0., 1)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutMotionFunc(motion)
    glutMouseFunc(mouse)
    glutTimerFunc(opt.refresh_rate, timer, 0)

    # alpha blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # depth buffer
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glDepthFunc(GL_LEQUAL)

    # fancy rendering
    glLineWidth(2.0)
    glShadeModel(GL_SMOOTH)

    init()
    print(USAGE)
    glutMainLoop()

if __name__ == "__main__":
    main()
