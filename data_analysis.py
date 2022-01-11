# Analysis of data
# Author: Edmund Dable-Heath
# analysing the data from the simulations

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_polar_bar(data, kl=False):
    # split up and tidy up data
    if kl:
        radii = data[:, 2]
    else:
        radii = data[:, 1]
    for i in range(len(radii)):
        if radii[i] == 0:
            radii[i] = 0
        else:
            radii[i] = 1/radii[i]
    theta = data[:, 0]
    width = 2*np.pi / len(data)
    colours = plt.cm.viridis(radii / 10.)

    ax = plt.subplot(111, projection='polar')
    ax.bar(theta, radii, width=width, bottom=0.0, color=colours, alpha=0.5)
    xT = plt.xticks()[0]
    xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
          r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    plt.xticks(xT, xL)
    plt.show()


def plot_both(data):
    # split up and tidy data:
    ratio_data = data[:, 1]
    kl_data = data[:, 2]
    for i in range(len(data)):
        if ratio_data[i] == 0:
            ratio_data[i] = 0
        else:
            ratio_data[i] = 1/ratio_data[i]
        if kl_data[i] == 0:
            kl_data[i] = 0
        else:
            kl_data[i] = 1/kl_data[i]
    theta = data[:, 0]
    width = 2*np.pi / len(data)
    ratio_colours = plt.cm.viridis(ratio_data / 10.)
    kl_colours = plt.cm.viridis(kl_data / 10.)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(polar=True))
    ax1[0, 0].bar(theta, ratio_data, width=width, bottom=0.0, color=ratio_colours, alpha=0.5)
    ax2[1, 1].bar(theta, kl_data, width=width, bottom=0.0, color=kl_colours, alpha=0.5)
    xT = plt.xticks()[0]
    # Label theta in radians
    xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
          r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    plt.xticks(xT, xL)
    plt.show()


def plot_bar(data):
    series = pandas.read_csv(data)
    series.columns = ['theta', 'gamma']
    series['gamma'] = 1/series['gamma']
    series.replace([np.inf], 0, inplace=True)
    print(series)
    series.plot(x='theta', y='gamma', style='k.')
    plt.show()


def four_by_four(dimension, index):
    # hnf dataframe
    hnf_data = pandas.read_csv('Theta_Results/dim_'+str(dimension)+'_HNF/results-'+str(dimension)+str(index)+'.csv')
    hnf_data.columns = ['theta', 'gamma']
    hnf_data['gamma'] = 1/hnf_data['gamma']
    hnf_data.replace([np.inf], 0, inplace=True)
    hnf_colours = plt.cm.viridis(hnf_data['gamma'] / 10.)

    # lll dataframe
    lll_data = pandas.read_csv('Theta_Results/dim_'+str(dimension)+'_LLL/results-'+str(dimension)+str(index)+'.csv')
    lll_data.columns = ['theta', 'gamma']
    lll_data['gamma'] = 1/lll_data['gamma']
    lll_data.replace([np.inf], 0, inplace=True)
    lll_colours = plt.cm.viridis(lll_data['gamma'] / 10.)

    # width of bars
    width = 2*np.pi / 1200

    # Setting up multiplot
    fig = plt.subplots(2, 2)

    # hnf polar chart
    hnf_polar = plt.subplot(221, projection='polar')
    hnf_polar.set_title('HNF Basis')
    hnf_polar.bar(hnf_data['theta'], hnf_data['gamma'], width=width, bottom=0.0, color=hnf_colours, alpha=0.5)
    yT = plt.xticks()[0]
    yL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
          r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    plt.xticks(yT, yL)
    # hnf_polar.xticks(xT, xL)

    # lll polar chart
    lll_polar = plt.subplot(222, projection='polar')
    lll_polar.set_title('LLL Basis')
    lll_polar.bar(lll_data['theta'], lll_data['gamma'], width=width, bottom=0.0, color=lll_colours, alpha=0.5)
    xT = plt.xticks()[0]
    xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
          r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    plt.xticks(xT, xL)

    # hnf bar/time series
    hnf_bar = plt.subplot(223)
    hnf_bar.set_xlabel('Radial Parameter')
    hnf_bar.set_ylabel('Figure of merit')
    hnf_bar.plot(hnf_data['theta'], hnf_data['gamma'], 'k.')

    # lll bar/time series
    lll_bar = plt.subplot(224)
    lll_bar.set_xlabel('Radial Parameter')
    lll_bar.set_ylabel('Figure of merit')
    lll_bar.plot(lll_data['theta'], lll_data['gamma'], 'k.')

    #Title
    # fig.suptitle('Radial and Series Plots of Data', fontsize=16)

    plt.show()


def plot_2_by_1(dimension, index, basis_type):
    # dataframe
    data = pandas.read_csv('Theta_Results/dim_'+str(dimension)+'_'+basis_type+'/results-'+str(dimension)+str(index)+'.csv')
    data.columns = ['theta', 'gamma']
    data['gamma'] = 1/data['gamma']
    data.replace([np.inf], 0, inplace=True)
    data_colours = plt.cm.viridis(data['gamma']/10.)

    # width of bars
    width = 2*np.pi / 1200

    # set up multiplot
    fig = plt.subplots(1, 2)

    # polar chart
    polar = plt.subplot(121, projection='polar')
    polar.bar(data['theta'], data['gamma'], width=width, bottom=0.0, color=data_colours, alpha=0.5)
    yT = plt.xticks()[0]
    yL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
          r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    plt.xticks(yT, yL)

    # time series chart
    series = plt.subplot(122)
    series.set_xlabel('Radial Parameter')
    series.set_ylabel('Figure of merit')
    series.plot(data['theta'], data['gamma'], 'k.')

    plt.show()


def plot_all_four(dimension):
    for i in range(32):
        print('HNF:')
        print(np.genfromtxt('Lattices/'+str(dimension)+'/' + str(i) + '/hnf.csv', delimiter=',', dtype=None))
        print('LLL:')
        print(np.genfromtxt('Lattices/'+str(dimension)+'/' + str(i) + '/lll.csv', delimiter=',', dtype=None))
        four_by_four(dimension, i)


def plot_all_two(dimension, basis_type):
    for i in range(32):
        print('HNF:')
        print(np.genfromtxt('Lattices/'+str(dimension)+'/' + str(i) + '/hnf.csv', delimiter=',', dtype=None))
        print('LLL:')
        print(np.genfromtxt('Lattices/'+str(dimension)+'/' + str(i) + '/lll.csv', delimiter=',', dtype=None))
        plot_2_by_1(dimension, i, basis_type)


def integer_to_lattice_plotter(basis, shortest_vector):
    def range_calc(latt_basis):
        dimension = latt_basis.shape[0]
        return dimension * math.log2(dimension) + math.log2(abs(np.linalg.det(latt_basis)))
    int_range = int(range_calc(basis))
    int_x, int_y = np.mgrid[-int_range:int_range, -int_range: int_range]
    z = np.zeros((2*int_range, 2*int_range))
    for i in range(int_x.shape[0]):
        for j in range(int_x.shape[0]):
            z[i][j] = np.linalg.norm(int_x[i][j]*basis[0] + int_y[i][j]*basis[1])

    # shortest vector algebraic solution
    shortest_ints = np.linalg.solve(basis.T, shortest_vector)

    #  plotting
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(int_x, int_y, z, cmap='viridis')
    ax.plot3D([-shortest_ints[0], -shortest_ints[0]],
              [-shortest_ints[1], -shortest_ints[1]],
              [0, np.max(z)], 'gray')
    ax.scatter(-shortest_ints[0], -shortest_ints[1], np.max(z))
    ax.plot3D([shortest_ints[0], shortest_ints[0]],
              [shortest_ints[1], shortest_ints[1]],
              [0, np.max(z)], 'gray')
    ax.scatter(shortest_ints[0], shortest_ints[1], np.max(z))
    plt.show()


if __name__ == "__main__":
    for i in range(32):
        sv = np.genfromtxt('Lattices/2/'+str(i)+'/sv.csv')
        integer_to_lattice_plotter(np.genfromtxt('Lattices/2/'+str(i)+'/hnf.csv', delimiter=',', dtype=None), sv)
        integer_to_lattice_plotter(np.genfromtxt('Lattices/2/'+str(i)+'/lll.csv', delimiter=',', dtype=None), sv)





