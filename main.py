import numpy as np
import pdb
from numpy import *
import scipy.io as sio
from numpy.linalg import inv
from numpy.linalg import norm
import numpy.linalg as lin
import os

from update_equations import *
from get_samples import *
from configuration import *


def GrouPS(*args, **kwargs):

    Iterations_number = 0
    Inner_iterations_number = 0
    reward_plot = np.zeros([1, max_iterations])
    check_if_converged = True
    check_if_converged2 = True

    W = np.empty([number_of_groups, 1], dtype=object)
    tau = np.empty([number_of_groups, 1], dtype=object)
    alpha = np.empty([number_of_groups, 1], dtype=object)
    initialM = np.empty([number_of_groups, 1], dtype=object)

    for m in range(0, number_of_groups):
        W[m][0] = np.random.randn(dimensions_per_group[m],
                                  latent_dimension_size)
        tau[m][0] = initial_Tau
        alpha[m] = initial_Alpha

    for m in range(0, number_of_groups):
        initialM[m][0] = np.zeros(
            [dimensions_per_group[m], orginial_feature_dimension_size])

    M = np.copy(initialM)
    if (load_the_latest_state == True):

        checkpoint = np.load('checkpoint.npy')
        M = checkpoint[0]
        W = checkpoint[1]
        tau = checkpoint[2]
        alpha = checkpoint[3]

    get_samples(
        W,
        M,
        tau,
        latent_dimension_size,
        dimensions_per_group,
        Time,
        rendering=1,
        nout=4)

    while check_if_converged:
        check_if_converged2 = True
        sum_of_rewards = 0

        ###################  Collect multiple samples with existing parameters ###############

        if sample_size == sample_size or Iterations_number == 0:
            Realization = np.empty([sample_size, 1], dtype=object)
            Z_temp = np.empty([sample_size, 1], dtype=object)
            for r in range(0, sample_size):
                Actions, Basisfunctions, Reward, Zvals = get_samples(
                    W,
                    M,
                    tau,
                    latent_dimension_size,
                    dimensions_per_group,
                    Time,
                    nout=4)
                Realization[r][0] = {
                    'Actions': Actions,
                    'Basis': Basisfunctions,
                    'Reward': Reward
                }
                Z_temp[r][0] = np.empty([Time, 1], dtype=object)

                for t in range(0, Time):
                    Z_temp[r][0][t][0] = {
                        'Mean': Zvals[:, t],
                        'Cov': np.eye(latent_dimension_size)
                    }
                    Z_temp[r][0][t][0]['Mean'] = Z_temp[r][0][t][0][
                        'Mean'].reshape(1, -1)

        reward_tmp = np.empty([sample_size, 1], dtype=object)

        for r in range(0, sample_size):
            reward_tmp[r][0] = np.sum(Realization[r][0]['Reward'])

        Idx = [
            i[0] for i in sorted(enumerate(reward_tmp), key=lambda x: -x[1])
        ]
        Best_samples_indices = np.array(Idx[0:best_size])
        sum_of_rewards = 0
        for r in Best_samples_indices.reshape(-1):
            sum_of_rewards = sum_of_rewards + np.sum(
                Realization[r][0]['Reward'])

        ################## Initialize various distributions #######################

        W_temp = np.empty([number_of_groups, 1], dtype=object)
        M_temp = np.empty([number_of_groups, 1], dtype=object)
        tau_temp = np.empty([number_of_groups, 1], dtype=object)
        alpha_temp = np.empty([number_of_groups, 1], dtype=object)

        for m in range(0, number_of_groups):
            W_temp[m][0] = np.empty([dimensions_per_group[m], 1], dtype=object)
            M_temp[m][0] = np.empty([dimensions_per_group[m], 1], dtype=object)
            alpha_temp[m][0] = np.empty(
                [latent_dimension_size, 1], dtype=object)
            for j in range(0, dimensions_per_group[m]):
                W_temp[m][0][j][0] = {
                    'Mean': W[m][0][j, :],
                    'Cov': np.eye(latent_dimension_size)
                }
                W_temp[m][0][j][0]['Mean'] = W_temp[m][0][j][0][
                    'Mean'].reshape(1, -1)

                M_temp[m][0][j][0] = {
                    'Mean': M[m][0][j, :],
                    'Cov': np.eye(latent_dimension_size)
                }
                M_temp[m][0][j][0]['Mean'] = M_temp[m][0][j][0][
                    'Mean'].reshape(1, -1)

            tau_temp[m][0] = {
                'A': tauA + np.dot(np.dot(0.5, dimensions_per_group[m]), Time),
                'B': tauB + np.dot(np.dot(0.5, dimensions_per_group[m]), Time)
            }
            for k in range(0, latent_dimension_size):
                alpha_temp[m][0][k][0] = {
                    'A': alphaA + dimensions_per_group[m] / 2,
                    'B': alphaB + dimensions_per_group[m] / 2
                }

        Inner_iterations_number = 0

        ############################### Start the iterations #######################

        while check_if_converged2:

            M_old_temp = np.empty([number_of_groups, 1], dtype=object)

            for m in range(0, number_of_groups):
                M_old_temp[m][0] = np.empty(
                    [dimensions_per_group[m], 1], dtype=object)

                for j in range(0, dimensions_per_group[m]):
                    M_old_temp[m][0][j] = {
                        'Mean': np.copy(M_temp[m][0][j][0]['Mean']),
                        'Cov': np.copy(M_temp[m][0][j][0]['Cov'])
                    }
                    M_old_temp[m][0][j][0]['Mean'] = M_old_temp[m][0][j][0][
                        'Mean'].reshape(1, -1)

            ####################### update M - Covariance ##############################

            M_temp = update_M_Covariance(
                Best_samples_indices, Realization, M_temp, tau_temp,
                orginial_feature_dimension_size, number_of_groups, sigma2_M,
                Time, sum_of_rewards, dimensions_per_group)

            ####################### update M - Mean ###################################

            M_temp = update_M_Mean(
                Best_samples_indices, Realization, M_temp, M, W_temp, Z_temp,
                tau_temp, orginial_feature_dimension_size, number_of_groups,
                sigma2_M, Time, sum_of_rewards, dimensions_per_group)

            ####################### update W - Covariance ##############################

            W_temp = update_W_Covariance(
                Best_samples_indices, Realization, M_temp, W_temp, Z_temp,
                alpha_temp, tau_temp, latent_dimension_size, number_of_groups,
                Time, sum_of_rewards, dimensions_per_group)

            ####################### update W - Mean ################################

            W_temp = update_W_Mean(Best_samples_indices, Realization, M_temp,
                                   W_temp, Z_temp, tau_temp,
                                   latent_dimension_size, number_of_groups,
                                   Time, sum_of_rewards, dimensions_per_group)

            ####################### update Z - Mean and Covariance ##############################

            Z_temp = update_Z_Mean_and_Covariance(
                Best_samples_indices, Realization, M_temp, W_temp, Z_temp,
                tau_temp, orginial_feature_dimension_size,
                latent_dimension_size, number_of_groups, Time,
                dimensions_per_group)

            ####################### update Tau B ##############################

            tau_temp = update_tau(
                Best_samples_indices, Realization, M_temp, W_temp, Z_temp,
                tau_temp, orginial_feature_dimension_size,
                latent_dimension_size, number_of_groups, Time, sum_of_rewards,
                dimensions_per_group, tauB)
            ####################### update alpha B ##############################

            alpha_temp, valueNorm_M_mean = update_alpha(
                M_temp, M_old_temp, W_temp, alpha_temp, alphaB,
                latent_dimension_size, number_of_groups, dimensions_per_group)

            check_if_converged2 = valueNorm_M_mean > 0.01
            print(Iterations_number, Inner_iterations_number, valueNorm_M_mean)

            if Inner_iterations_number >= max_inner_iterations:
                check_if_converged2 = 0
            else:
                Inner_iterations_number = Inner_iterations_number + 1

        ############################# Copy the learnt values of M,W,alpha and tau ###################

        for xx in range(0, number_of_groups):
            alpha[xx][0] = np.empty([latent_dimension_size, 1], dtype=object)

        for m in range(0, number_of_groups):
            for j in range(0, dimensions_per_group[m]):
                M[m][0][j] = M_temp[m][0][j][0]['Mean'][0]
                W[m][0][j] = np.dot(anti_convergence_factor,
                                    W_temp[m][0][j][0]['Mean'])[0]

            tau[m][0] = lin.solve(
                np.array([anti_convergence_factor]).reshape(1, -1),
                np.array((tau_temp[m][0]['A'] / tau_temp[m][0]['B'])).reshape(
                    1, -1))[0][0]

            for k in range(0, latent_dimension_size):
                alpha[m][0][k] = alpha_temp[m][0][k][0]['A'] / alpha_temp[m][
                    0][k][0]['B']

        Iterations_number = Iterations_number + 1
        check_if_converged = Iterations_number < max_iterations

        checkpoint = np.array([M, W, tau, alpha])
        np.save('checkpoint.npy', checkpoint)

        if (Iterations_number % 5 == 0):
            os.mkdir(str(Iterations_number))
            np.save('./' + str(Iterations_number) + '/checkpoint.npy',
                    checkpoint)

        get_samples(
            W,
            M,
            tau,
            latent_dimension_size,
            dimensions_per_group,
            Time,
            rendering=1,
            nout=4)

        print('Iteration :', Iterations_number)

    reward_over_time = np.copy(reward_plot)
    return M, W, tau, alpha, reward_over_time


if __name__ == '__main__':

    np.random.seed(10)
    M, W, tau, alpha, reward_over_time = GrouPS()
