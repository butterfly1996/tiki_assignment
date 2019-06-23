import numpy as np
import matplotlib.pyplot as plt
from src.generate_data import gen_data

# function that calculate the probability of a input belong to each classes
def prob(x1, x2, weights, means, cov):
    total_res = []
    q = np.linalg.inv(cov)
    for x in zip(x1, x2):
        res = 0
        for i in range(len(means)):
            _x = x - means[i]
            res += (np.exp(-0.5 * np.dot(np.dot(_x.transpose(), q), _x)))*weights[i]
            res = np.array(res)
        total_res.append(res)
    total_res = np.array(total_res)
    return total_res


def solution(
    num_points=100,
    n_gaussians=5,
    min_mean_1=-1,
    max_mean_1=0,
    min_mean_2=0,
    max_mean_2=1):
    cov = np.array([[1, 0], [0, 1]])  # assume that X and Y are independent
    # generrate weights for mixtured gauss
    weights = np.random.uniform(1, 100, (2, n_gaussians))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    # generate means
    means = [
        np.random.uniform(min_mean_1, max_mean_1, (n_gaussians, 2)),
        np.random.uniform(min_mean_2, max_mean_2, (n_gaussians, 2))]
    points = gen_data(num_points, weights, means, cov)
    if points is None:
        print("Fail to generate data")
        return None
    # plot points
    plt.figure()
    for i, p in enumerate(points):
        if i == 0:
            c = 'red'
            marker = 'o'
        else:
            c = 'blue'
            marker = 'x'
        x, y = p.transpose()
        plt.scatter(x, y, c=c, marker=marker)
    # compute Bayes Boundary
    input_point = np.concatenate(points)
    x_min, x_max = min(input_point[:, 0]), max(input_point[:, 0])
    y_min, y_max = min(input_point[:, 1]), max(input_point[:, 1])
    X1, X2 = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
    x1 = X1.ravel()
    x2 = X2.ravel()
    i = (prob(x1, x2, weights[0], means[0], cov) - prob(x1, x2, weights[1], means[1], cov)).reshape(X1.shape)
    plt.contour(X1, X2, i, levels=[0])
    # calculate accuracy
    acc = 0

    for label in range(2):
        classification_result = prob(points[label][:, 0], points[label][:, 1], weights[0], means[0], cov) -\
                                prob(points[label][:, 0], points[label][:, 1], weights[1], means[1], cov)
        classification_result = classification_result*(1 - 2*label)
        acc += np.sum(classification_result >= 0, axis=0)
    print("accuracy: %0.4f%%" % (100*acc/(2*num_points)))
    plt.show()


if __name__ == '__main__':
    solution()
