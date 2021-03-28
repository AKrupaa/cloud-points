from _csv import writer

from scipy.stats import norm
from typing import Literal
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

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

