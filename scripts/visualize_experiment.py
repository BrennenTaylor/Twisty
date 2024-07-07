import json
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def plot_camera(ax, config):
    cam = config['cameraPos']
    eye_pos = cam['eyePos']
    look_at = cam['lookAt']
    up = cam['up']

    # Plot the camera position
    ax.scatter(eye_pos['x'], eye_pos['y'], eye_pos['z'], color='r', s=100, label='Camera Position')

    # Plot the camera's view direction
    ax.quiver(eye_pos['x'], eye_pos['y'], eye_pos['z'],
              look_at['x'] - eye_pos['x'], look_at['y'] - eye_pos['y'], look_at['z'] - eye_pos['z'],
              length=1.0, color='r')

    # Plot the up direction
    ax.quiver(eye_pos['x'], eye_pos['y'], eye_pos['z'],
              up['x'], up['y'], up['z'],
              length=1.0, color='g', label='Up Direction')

def plot_emitters(ax, emitters):
    for i, emitter in enumerate(emitters):
        pos = emitter['position']
        dir = emitter['direction']
        
        # Plot emitter position
        ax.scatter(pos['x'], pos['y'], pos['z'], s=100, label=f'Emitter {i+1} Position')
        
        # Plot emitter direction
        ax.quiver(pos['x'], pos['y'], pos['z'],
                  dir['x'], dir['y'], dir['z'],
                  length=1.0, label=f'Emitter {i+1} Direction')

# def plot_frame(ax, config):
#     nci = config['noisyCircleAngleIntegration']
#     frame_center = [nci['startX'], nci['startY'], 0]
#     frame_length = nci['frameLength']
    
#     # Calculate frame corners
#     half_len = frame_length / 2.0
#     corners = [
#         [frame_center[0] - half_len, frame_center[1] - half_len, frame_center[2]],
#         [frame_center[0] - half_len, frame_center[1] + half_len, frame_center[2]],
#         [frame_center[0] + half_len, frame_center[1] + half_len, frame_center[2]],
#         [frame_center[0] + half_len, frame_center[1] - half_len, frame_center[2]],
#     ]
    
#     corners = np.array(corners)
    
#     # Plot frame
#     ax.plot([corners[0][0], corners[1][0]], [corners[0][1], corners[1][1]], [corners[0][2], corners[1][2]], 'b-')
#     ax.plot([corners[1][0], corners[2][0]], [corners[1][1], corners[2][1]], [corners[1][2], corners[2][2]], 'b-')
#     ax.plot([corners[2][0], corners[3][0]], [corners[2][1], corners[3][1]], [corners[2][2], corners[3][2]], 'b-')
#     ax.plot([corners[3][0], corners[0][0]], [corners[3][1], corners[0][1]], [corners[3][2], corners[0][2]], 'b-')

def plot_frustum(ax, config):
    cam = config['cameraPos']
    eye_pos = np.array([cam['eyePos']['x'], cam['eyePos']['y'], cam['eyePos']['z']])
    look_at = np.array([cam['lookAt']['x'], cam['lookAt']['y'], cam['lookAt']['z']])
    up = np.array([cam['up']['x'], cam['up']['y'], cam['up']['z']])
    fov = cam['fovDegrees']
    aspect = cam['aspectRatio']
    distance = config['noisyCircleAngleIntegration']['distanceFromPlane']

    # Compute forward, right, and up vectors
    forward = look_at - eye_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_vector = np.cross(right, forward)

    # Compute the half height and width of the frame plane
    half_height = np.tan(np.radians(fov) / 2) * distance
    half_width = half_height * aspect

    # Compute corners of the frame plane
    center = eye_pos + forward * distance
    top_left = center + up_vector * half_height - right * half_width
    top_right = center + up_vector * half_height + right * half_width
    bottom_left = center - up_vector * half_height - right * half_width
    bottom_right = center - up_vector * half_height + right * half_width

    # Define the edges of the frustum
    edges = [
        [eye_pos, top_left], [eye_pos, top_right], [eye_pos, bottom_left], [eye_pos, bottom_right],
        [top_left, top_right], [top_right, bottom_right], [bottom_right, bottom_left], [bottom_left, top_left]
    ]

    for edge in edges:
        ax.plot3D(*zip(*edge), color="purple")

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file> [--show-frustum]")
        return
    
    config_file = sys.argv[1]
    show_frustum = '--show-frustum' in sys.argv

    config = load_config(config_file)['experiment']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plot_camera(ax, config)
    plot_emitters(ax, config['emitters'])
    # plot_frame(ax, config)

    if show_frustum:
        plot_frustum(ax, config)

    set_axes_equal(ax)
    ax.view_init(elev=90, azim=-90)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
