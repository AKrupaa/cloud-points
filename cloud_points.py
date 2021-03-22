from _csv import writer

from scipy.stats import norm
from typing import Literal

Options = Literal["flat_horizontal", 'flat_vertical', 'tube', 'random']


def generate_points(num_points: int = 2000, option: Options = 'random'):
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


if __name__ == '__main__':
    # main()
    cloud_points = generate_points(10000, 'tube')
    with open('Lidar.csv', 'w', encoding='utf-8', newline='\n') as csv_file:
        csv_file_writer = writer(csv_file)
        for point in cloud_points:
            csv_file_writer.writerow(point)
    print(f'Done')
