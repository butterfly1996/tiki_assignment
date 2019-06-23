import numpy as np
import matplotlib.pyplot as plt


def gen_data(
	num_points=100,
	weights=None,
	means=None,
	cov=np.array([[1, 0], [0, 1]])):
	points = []
	for label in range(len(means)):
		mean = means[label]
		weight = weights[label]
		p = []
		for i in range(num_points):
			one_hot_vector = np.random.multinomial(1, weight)
			idx = np.where(one_hot_vector == 1)[0][0]
			x = np.random.multivariate_normal(mean[idx], cov)
			p.append(x)
		points.append(p)
	return np.array(points)


if __name__ == '__main__':
	num_points = 100
	n_gaussians = 5
	min_mean_1 = -1
	max_mean_1 = 0
	min_mean_2 = 0
	max_mean_2 = 1
	cov = np.array([[1, 0], [0, 1]])  # assume that X and Y are independent
	# generrate weights for mixtured gauss
	weights = np.random.uniform(1, 100, (2, n_gaussians))
	weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
	# generate means
	means = [
		np.random.uniform(min_mean_1, max_mean_1, (n_gaussians, 2)),
		np.random.uniform(min_mean_2, max_mean_2, (n_gaussians, 2))]
	points = gen_data(num_points, weights, means, cov)
	plt.figure()
	# plot points
	for i, p in enumerate(points):
		if i == 0:
			c = 'red'
			marker = 'o'
		else:
			c = 'blue'
			marker = 'x'
		x, y = p.transpose()
		plt.scatter(x, y, c=c, marker=marker)
	plt.show()
