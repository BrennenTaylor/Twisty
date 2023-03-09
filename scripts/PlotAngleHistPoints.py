import sys
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import tkinter
from tkinter import filedialog
import os
import random

root = tkinter.Tk()
root.withdraw() #use to hide tkinter window

def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(parent=root, initialdir="C:/", title='Please select a directory')
    if len(tempdir) > 0:
        print ("You chose: %s" % tempdir)
    return tempdir

# Parse Command line argument integer representing number of items to load
if len(sys.argv) != 2:
    print ("Error: Please enter a single integer as a command line argument")
    exit()
num_items = int(sys.argv[1])

file_path_variable = search_for_file_path()
print ("\nfile_path_variable = ", file_path_variable)
angle_file = open(file_path_variable + "/angle.dat", "r")

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,128)

ax.set_ylim(0,128)

ax.set_zlim(0,128)

angle_list = []
for line in angle_file:
    # If 4th entry is at least one, the color is red
    # Otherwise, the color is blue
    # Load x, y, z coordinates and add to a list
    if line.split()[3] == '1':
        angle_list.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])

# Randomly shuffle the list
random.shuffle(angle_list)
angle_list_subset = angle_list[:num_items]

# Select the first 100 points
print (len(angle_list))
print (len(angle_list_subset))
# Plot the subset
ax.scatter([x[0] for x in angle_list_subset], [x[1] for x in angle_list_subset], [x[2] for x in angle_list_subset], c='r', marker='o', alpha=0.1)

plt.show()
