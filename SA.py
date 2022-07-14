#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:57:07 2022

@author: henk
"""
import numpy as np

from estimatorClass import SF, IPA
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera, norm

def compute_convergence_rate(constant_epsilon, epsilon, gamma, boundary_function, delta, n_simulations, n, alpha, variance, f, batching, batch_size):
    """ Boundary function. """
    def boundary(x):
        """ We require delta < 1. """
        if x <= delta:
            return delta
        elif x >= 1 / delta:
            return 1 / delta
        else: 
            return x   
    
    def SA(theta_n, epsilon, Y_n, j, gamma):
        """ Use constant or decreasing epsilon."""
        if constant_epsilon == False:
            epsilon = (j+1)**(-gamma)
        
        """ Use boundary function or not."""
        if boundary_function == False:
            return theta_n - epsilon * Y_n
        else:
            return boundary(theta_n - epsilon * Y_n)

    theta_n_IPA = np.empty((n, n_simulations))
    theta_n_SF  = np.empty((n, n_simulations))
    iteration = [int((n / 4) * i) for i in range(1,5)]
    fig, ax = plt.subplots(4, 4, constrained_layout = True)
    
    
    for i in range(n_simulations):
        if i % 50 == 0:
            print(i)
        """ Initialize theta_0 randomly."""
        thetaIPA = np.random.random() * (1 / delta)
        thetaSF  = np.random.random() * (1 / delta)
    
        for j in range(n):
            IPAn = IPA(thetaIPA, alpha, variance)
            SFn  = SF(thetaSF, alpha,variance)
            
            if f == 0 and batching == True:
                meanIPA = np.mean([IPAn.norm(thetaIPA, alpha, variance) for i in range(batch_size)])
                meanSF  = np.mean([SFn.norm(thetaSF, alpha, variance)  for i in range(batch_size)])
                
                thetaIPA = SA(thetaIPA, epsilon, meanIPA, j, gamma)
                thetaSF  = SA(thetaSF,  epsilon, meanSF, j, gamma)
                
            elif f == 0 and batching == False:
                thetaIPA = SA(thetaIPA, epsilon, IPAn.norm(thetaIPA, alpha, variance), j, gamma)
                thetaSF  = SA(thetaSF, epsilon, SFn.norm(thetaSF, alpha, variance), j, gamma)
            
            elif f == 1 and batching == True:
                meanIPA = np.mean([IPAn.exp(thetaIPA, alpha) for i in range(batch_size)])
                meanSF  = np.mean([SFn.exp(thetaSF, alpha)  for i in range(batch_size)])
                
                thetaIPA = SA(thetaIPA, epsilon, meanIPA, j, gamma)
                thetaSF  = SA(thetaSF,  epsilon, meanSF, j, gamma)
            
            elif f == 1 and batching == False:
                thetaIPA = SA(thetaIPA, epsilon, IPAn.exp(thetaIPA, alpha), j, gamma)
                thetaSF  = SA(thetaSF, epsilon, SFn.exp(thetaSF, alpha), j, gamma)
            
            elif f == 2 and batching == True:
                meanIPA = np.mean([IPAn.mining(thetaIPA) for i in range(batch_size)])
                meanSF  = np.mean([SFn.mining(thetaSF)  for i in range(batch_size)])
                
                thetaIPA = SA(thetaIPA, epsilon, meanIPA, j, gamma)
                thetaSF  = SA(thetaSF,  epsilon, meanSF, j, gamma)
            
            elif f == 2 and batching == False:
                thetaIPA = SA(thetaIPA, epsilon, IPAn.mining(thetaIPA), j, gamma)
                thetaSF  = SA(thetaSF, epsilon, SFn.mining(thetaSF), j, gamma) 
            
            theta_n_IPA[j][i] = thetaIPA
            theta_n_SF[j][i]  = thetaSF
            
        if i % 2 == 0:   
            ax[0][0].plot(np.array(theta_n_IPA).T[i], color = 'lightblue', alpha = 0.1)
            ax[0][1].plot(np.array(theta_n_SF).T[i],  color = 'orange', alpha = 0.1)
            
    
    X = np.linspace(start=0.00001,stop=n,num=n)
    
    mean_IPA = np.array([np.mean(i) for i in theta_n_IPA])
    var_IPA = [np.var(i) for i in theta_n_IPA]
    skew_IPA = [skew(i) for i in theta_n_IPA]
    #CI_IPA = np.array([1.96 * np.sqrt(var_IPA[i]) / np.sqrt(n_simulations) for i in range(n)])
    conv_IPA = [np.log2(var_IPA[i*2] / var_IPA[i]) for i in range(int(n/2))]
    second_moment_IPA = [np.mean(i**2) for i in np.array(theta_n_IPA)]
    kurt_IPA = [kurtosis(i) for i in theta_n_IPA]
    JB_IPA = [(n_simulations / 6) * (skew_IPA[i]**2 + 0.25*(kurt_IPA[i]-3)**2) for i in range(n)]
    JB_IPA2 = [jarque_bera(i)[0] for i in theta_n_IPA]

    
    mean_SF = np.array([np.mean(i) for i in theta_n_SF])
    var_SF = [np.var(i) for i in theta_n_SF]
    skew_SF = [skew(i) for i in theta_n_SF]   
    #CI_SF = np.array([1.96 * np.sqrt(var_SF[i]) / np.sqrt(n_simulations) for i in range(n)])
    conv_SF = [np.log2(var_SF[i*2] / var_SF[i]) for i in range(int(n/2))]
    second_moment_SF = [np.mean(i**2) for i in theta_n_SF]
    kurt_SF = [kurtosis(i) for i in theta_n_SF]
    JB_SF = [(n_simulations / 6) * (skew_IPA[i]**2 + 0.25*(kurt_SF[i]-3)**2) for i in range(n)]
    JB_SF2 = [jarque_bera(i)[0] for i in theta_n_SF]
    
    
    ax[0][0].plot(mean_IPA, label = 'mean IPA', color = 'blue')
    #ax[0][0].fill_between(X, mean_IPA - CI_IPA, mean_IPA + CI_IPA, color = 'blue', alpha = 0.1)
    
    ax[0][1].plot(mean_SF, label = 'mean SF', color = 'r')
    #ax[0][1].fill_between(X, mean_SF - CI_SF, mean_SF + CI_SF, color = 'orange', alpha = 0.1)
    
    ax[1][0].plot(var_IPA, label = 'var', color = 'blue')
    ax[1][1].plot(var_SF, label = 'var', color = 'orange')
    
    ax[1][0].plot(second_moment_IPA, label = '2nd mom.', color = 'blueviolet')
    ax[1][1].plot(second_moment_SF, label = '2nd mom.', color = 'brown')

    ax[2][0].plot([0 for i in X], color = 'grey', linewidth = 1)
    ax[2][1].plot([0 for i in X], color = 'grey', linewidth = 1)
    
    ax[2][0].scatter(X, skew_IPA, label = 'skew', color = 'lightgreen', s=10)
    ax[2][1].scatter(X, skew_SF, label = 'skew', color = 'lightgreen', s=10)
    
    ax[2][0].scatter(X[:len(X)//2], conv_IPA, label = 'conv. rate', color = 'lightpink', s=10)
    ax[2][1].scatter(X[:len(X)//2], conv_SF, label = 'conv. rate', color = 'lightpink', s=10)
    
    ax[3][0].scatter(X, JB_IPA2, label = 'JB', s=10, color = 'blue')
    ax[3][1].scatter(X, JB_SF2, label = 'JB', s=10, color = 'orange')
    
    
    
    """ Plot the  first order Edgeworth expansion of the estimators."""
    def H2(x):
        return x**2 - 1
    
    Y = np.linspace(-4,4, n_simulations)
    Exact_cdf = norm.cdf(Y)
    
    for i in range(4):
        """ Boxplots of theta_n """
        ax[i][2].hist([theta_n_IPA[iteration[i]-1], theta_n_SF[iteration[i]-1]], 
                      bins = 50,
                      label = ['IPA', 'SF'], 
                      color = ['blue', 'orange'])
        ax[i][2].set_title(r'$\theta_{{n}}$ at n={0}'.format(iteration[i]))
        ax[i][2].set_ylabel('Frequency')
        
        """ First order Edgeworth expansion """
        # We also have to normalize our data
        normal_IPA = (theta_n_IPA[iteration[i]-1] - mean_IPA[iteration[i]-1]) / var_IPA[iteration[i]-1]
        normal_SF  = (theta_n_SF[iteration[i]-1] - mean_SF[iteration[i]-1]) / var_SF[iteration[i]-1]
        
        skew_IPA = skew(normal_IPA)
        skew_SF = skew(normal_SF)
        
        print(f'IPA skew is {skew_IPA} at {iteration[i]}')
        print(f'SF skew  is {skew_SF} at {iteration[i]}')
        
        IPA_cdf = norm.cdf(Y) - (norm.pdf(Y) * ((skew_IPA * H2(Y)) / (6 * np.sqrt(n_simulations))))
        SF_cdf = norm.cdf(Y) - (norm.pdf(Y) * ((skew_SF * H2(Y)) / (6 * np.sqrt(n_simulations))))
        
        ax[i][3].plot(Y,Exact_cdf - Exact_cdf, linewidth = 1, color = 'grey')  
        ax[i][3].plot(Y,Exact_cdf - IPA_cdf, linewidth = 1, label = 'IPA', color = 'blue')
        ax[i][3].plot(Y,Exact_cdf - SF_cdf, linewidth = 1, label = 'SF', color = 'orange')
        ax[i][3].set_title(f'Edgeworth at n={iteration[i]}')
        ax[i][3].set_ylabel('Difference')
        
        
        
        
        
        
        
    
    ax[0][0].set_title(r'Path of $\theta_{n}^{IPA}$')
    ax[0][1].set_title(r'Path of $\theta_{n}^{SF}$')
    ax[1][0].set_title(r'Variance, second moment at $\theta_{n}^{IPA}$')
    ax[1][1].set_title(r'Variance, second moment at $\theta_{n}^{SF}$')
    ax[1][0].set_yscale('log')
    ax[1][1].set_yscale('log')
    ax[3][0].set_yscale('log')
    ax[3][1].set_yscale('log')
    ax[1][0].legend()
    ax[1][1].legend()
    ax[2][0].set_title(r"""Skewness and convergence rate at $\theta_{{n}}^{{IPA}}$""")
    ax[2][1].set_title(r"""Skewness and convergence rate at $\theta_{{n}}^{{SF}}$""")
    ax[3][0].set_title('JB statistic of IPA at step n')
    ax[3][1].set_title('JB statistic of SF at step n.')
    ax[2][0].legend()

    ax[3][0].set_xlabel('n')
    ax[3][1].set_xlabel('n')
    ax[3][2].set_xlabel(r'$\theta_{n}$')
    ax[3][3].set_xlabel('x')
    ax[3][0].set_ylabel('JB')
    
    ax[0][0].set_ylabel(r'$\theta_{n}$')
    ax[1][0].set_ylabel(r'$\theta_{n}$')
    ax[2][0].set_ylabel(r'$\theta_{n}$')
    
    plt.legend()
    plt.suptitle(f'{n_simulations} simulations with {constant_epsilon} constant epsilon={epsilon} and {boundary_function} use of boundary fn \n with delta={delta}, alpha = {alpha} and {batching} batching with batch size {batch_size}.  Variance={variance}, f={f}, IPA is blue and SF is red.')
    
    print('-----------------------------------')
    print('convergence rate of IPA')
    print(conv_IPA)
    print('-----------------------------------')
    print('convergence rate of SF')
    print(conv_SF)
    print('-----------------------------------')
    print('JB values of IPA')
    print(JB_IPA)
    print('-------------------------------------')
    print('JB values of jarq_bar.test of IPA')
    print(JB_IPA2)
    print('--------------------------------------')
    print('p values of jarq_bar.test of IPA')
    print([jarque_bera(i)[1] for i in theta_n_IPA])
    print('-------------------------------------')
    print('JB values of SF')
    print(JB_SF)
    print('-------------------------------------')
    print('JB values of jarq_bar.test of SF')
    print(JB_SF2)
    print('--------------------------------------')
    print('p values of jarq_bar.test of SF')
    print([jarque_bera(i)[1] for i in theta_n_SF])
    print('--------------------------------------')
    print('variance of IPA')
    print(var_IPA)
    print('--------------------------------------')
    print('variance of SF')
    print(var_SF)
    
    return plt.show()




compute_convergence_rate(constant_epsilon= False, 
                         epsilon = 0.1, 
                         gamma = .75,
                         boundary_function = True, 
                         delta = 1/200, 
                         n_simulations = 5000, 
                         n = 100, 
                         alpha = 100,
                         variance = 9,
                         f = 1, 
                         batching = True, 
                         batch_size = 500)









































