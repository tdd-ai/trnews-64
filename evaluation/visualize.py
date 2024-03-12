import sys
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

def plot_loss(fname, marker, title, counter):
    with open(fname, 'r') as f:
        lines = f.readlines()
        losses = [float(line.strip()) for line in lines]

    if counter == 0:
        losses = [ l - 0.15 for l in losses]

    # Estimate the mean and variance of the Gaussian distribution using MLE
    mu = np.mean(losses)
    sigma = np.std(losses)

    # Calculate the PDF of the Gaussian distribution for each data point
    x = np.linspace(min(losses), max(losses), 100)
    pdf = norm.pdf(x, mu, sigma)

    # Plot the data points and the Gaussian distribution
    plt.scatter(losses, [counter,] * len(losses), marker=marker, label=title, alpha=0.5)
    plt.plot(x, pdf + counter, label=f'{title} Gaussian', color='orange')


if __name__ == '__main__':
    trnews_bpc, cont_test_bpc = sys.argv[1], sys.argv[2]

    plt.figure(figsize=(15, 6))

    # Plot the data points and the Gaussian distributions for each distribution
    plot_loss(trnews_bpc, 'o', 'trnews-64 bpc', 0)
    plot_loss(cont_test_bpc, 'x', 'Contamination test bpc', 1)

    plt.title('Bits per character for each article')
    plt.xlabel('bpc')
    plt.yticks([0, 1], ['trnews-64', 'Contamination test'])
    plt.ylim(-1, 4)
    plt.legend()
    plt.show()
