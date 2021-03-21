from _csv import writer

from scipy.stats import norm


def generate_points(num_points: int = 2000):
    distribution_x = norm(0, 20)
    distribution_y = norm(0, 20)
    distribution_z = norm(0, 20)

    num_points = 2000
    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


if __name__ == '__main__':
    # main()
    cloud_points = generate_points(2000)
    with open('Lidar.csv', 'w', encoding='utf-8', newline='\n') as csv_file:
        csv_file_writer = writer(csv_file)
        for point in cloud_points:
            csv_file_writer.writerow(point)
    print(f'Done')
