from _csv import writer

from scipy.stats import norm
from typing import Literal
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random
import math
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pyransac3d as pyrsc

Options = Literal["flat_horizontal", 'flat_vertical', 'tube', 'random']


def generate_points(num_points: int = 1000, option: Options = 'random'):
    if option == 'flat_horizontal':
        distribution_x = norm(0, 50)
        distribution_y = norm(0, 70)
        distribution_z = norm(0, 0)
    elif option == 'flat_vertical':
        distribution_x = norm(0, 50)
        distribution_y = norm(0, 0)
        distribution_z = norm(0, 90)
    elif option == 'tube':
        distribution_x = norm(0, 50)
        distribution_y = distribution_x
        distribution_z = norm(0, 90)
    else:
        distribution_x = norm(0, 50)
        distribution_y = norm(0, 50)
        distribution_z = norm(0, 90)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


def read_from_csv(file_name):
    with open(file=file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for x, y, z in reader:
            yield float(x), float(y), float(z)


def k_means_algorithm(list_of_read_points):
    n_clusters = 3

    X = np.array(list_of_read_points)

    k_means = KMeans(n_clusters=n_clusters)
    k_means = k_means.fit(X)
    labels = k_means.predict(X)

    # 3 plaszczyzny, czerwona, zielona, niebieska
    red = labels == 0
    green = labels == 1
    blue = labels == 2

    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(projection='3d')
    ax_2.scatter(X[red, 0], X[red, 1], X[red, 2], marker='o')
    ax_2.scatter(X[green, 0], X[green, 1], X[green, 2], marker='^')
    ax_2.scatter(X[blue, 0], X[blue, 1], X[blue, 2], marker='x')
    plt.show()


def ransac_algorithm(list_of_read_points, max_iters, threshold):
    while max_iters:
        max_iters -= 1
        inliers = []
        # three random points
        while len(inliers) < 3:
            random_index = random.randint(0, len(list_of_read_points) - 1)
            inliers.append(random_index)

        print(inliers)

        x1 = 0
        x2 = 0
        x3 = 0
        y1 = 0
        y2 = 0
        y3 = 0
        z1 = 0
        z2 = 0
        z3 = 0

        x1 = list_of_read_points[inliers[0], 0]
        y1 = list_of_read_points[inliers[0], 1]
        z1 = list_of_read_points[inliers[0], 2]

        x2 = list_of_read_points[inliers[1], 0]
        y2 = list_of_read_points[inliers[1], 1]
        z2 = list_of_read_points[inliers[1], 2]

        x3 = list_of_read_points[inliers[2], 0]
        y3 = list_of_read_points[inliers[2], 1]
        z3 = list_of_read_points[inliers[2], 2]

        # Plane Equation --> ax + by + cz + d = 0

        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
        plane_length = max(0.1, math.sqrt(a * a + b * b + c * c))

        pass
        # for point in list_of_read_points:
        #     if(point)


def k_mean_lib(data):
    clustering = DBSCAN(eps=15, min_samples=10).fit(data)
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)


if __name__ == '__main__':
    # main()
    flat_horizontal = generate_points(1000, 'flat_horizontal')
    flat_vertical = generate_points(1000, 'flat_vertical')
    tube = generate_points(100, 'tube')
    with open('Lidar.csv', 'w', encoding='utf-8', newline='\n') as csv_file:
        csv_file_writer = writer(csv_file)
        for point in flat_horizontal:
            csv_file_writer.writerow(point)
        for point in flat_vertical:
            csv_file_writer.writerow(point)
        for point in tube:
            csv_file_writer.writerow(point)
    print(f'Done')

    # read data
    list_of_read_points_from_csv = list(read_from_csv("Lidar.csv"))

    # studying
    X, Y, Z = zip(*list_of_read_points_from_csv)
    # print(f'{X}')
    # print(f'{Y}')
    # print(f'{Z}')

    #     plot 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X, Y, Z)
    # plt.show()

    # ########################################################
    # ########################################################
    k_means_algorithm(list_of_read_points_from_csv)
    # ########################################################
    # ########################################################
    X = np.array(list_of_read_points_from_csv)
    # ########################################################
    # ########################################################
    # ransac_algorithm(X, 10, 0.1)
    # ########################################################
    # ########################################################
    k_mean_lib(X)
    # ########################################################
    # ########################################################
    # points = load_points(.)  # Load your point cloud as a numpy array (N, 3)
    points = X

    # Example 1 - Planar RANSAC
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(points, 0.01)
    #     Results in the plane equation Ax+By+Cz+D: [1, 0.5, 2, 0]

    # Example 2 - Spherical RANSAC
    # Loading a noisy sphere's point cloud with r = 5 centered in 0 we can use the following code:

    # points = load_points(.) # Load your point cloud as a numpy array (N, 3)
    points = X
    sph = pyrsc.Sphere()
    center, radius, inliers = sph.fit(points, thresh=0.4)

    # center: [0.010462385575072288, -0.2855090643954039, 0.02867848979091283]
    # radius: 5.085218633039647
    # ########################################################
    # ########################################################

#     problemy cz. 1
# https://github.com/intel-isl/Open3D/issues/979
# $ pip install open3d
# ERROR: Could not find a version that satisfies the requirement open3d (from versions: none)
# ERROR: No matching distribution found for open3d

# Traceback (most recent call last):
#   File "E:\workplace\XXXX\cloud_points.py", line 13, in <module>
#     import pyransac3d as pyrsc
#   File "D:\Program Files\Python39\lib\site-packages\pyransac3d\__init__.py", line 4, in <module>
#     from .plane import Plane
#   File "D:\Program Files\Python39\lib\site-packages\pyransac3d\plane.py", line 1, in <module>
#     import open3d as o3d
#   File "D:\Program Files\Python39\lib\site-packages\open3d\__init__.py", line 13, in <module>
#     from open3d.win32 import *
#   File "D:\Program Files\Python39\lib\site-packages\open3d\win32\__init__.py", line 11, in <module>
#     globals().update(importlib.import_module('open3d.win32.64b.open3d').__dict__)
#   File "D:\Program Files\Python39\lib\importlib\__init__.py", line 127, in import_module
#     return _bootstrap._gcd_import(name[level:], package, level)
#   File "D:\Program Files\Python39\lib\site-packages\open3d\win32\64b\__init__.py", line 7, in <module>
#     globals().update(importlib.import_module('open3d.win32.64b.open3d').__dict__)
#   File "D:\Program Files\Python39\lib\importlib\__init__.py", line 127, in import_module
#     return _bootstrap._gcd_import(name[level:], package, level)
# ImportError: DLL load failed while importing open3d: Nie można odnaleźć określonego modułu.
#
# Process finished with exit code 1
