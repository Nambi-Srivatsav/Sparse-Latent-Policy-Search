import numpy as np
import pdb
from numpy import *
from get_samples import *
import scipy.io as sio
from numpy.linalg import inv
from numpy.linalg import norm
import numpy.linalg as lin
import os


def update_M_Covariance(Best_samples_indices,Realization,M_temp,tau_temp,orginial_feature_dimension_size,number_of_groups,sigma2_M,Time,sum_of_rewards,dimensions_per_group):
    
    Sigma_Phi=np.zeros([orginial_feature_dimension_size,orginial_feature_dimension_size])
            
    for r in Best_samples_indices.reshape(-1):
        Basis=Realization[r][0]['Basis']
        reward=Realization[r][0]['Reward'][0]
        for t in range(0,Time):
            Phi=Basis[:,t]
            Phi = Phi.reshape(-1,1)
            Sigma_Phi=Sigma_Phi + multiply((np.dot(Phi,Phi.T)) / (np.dot(Phi.T,Phi)),reward[t])
        
    Sigma_Phi=Sigma_Phi / sum_of_rewards

    for m in range(0,number_of_groups):
        tau_Ew=(tau_temp[m][0]['A'] / tau_temp[m][0]['B'])
        Sigma= inv((np.dot(Sigma_Phi,tau_Ew)) + np.eye(orginial_feature_dimension_size) / sigma2_M)
        for j in range(0,dimensions_per_group[m]):
            M_temp[m][0][j][0]['Cov'] = np.copy(Sigma)
            

    return M_temp
                    



def update_M_Mean(Best_samples_indices,Realization,M_temp,M,W_temp,Z_temp,tau_temp,orginial_feature_dimension_size,number_of_groups,sigma2_M,Time,sum_of_rewards,dimensions_per_group):
    
    for m in range(0,number_of_groups):
        startDim = sum(dimensions_per_group[:m])

        for j in range(0,dimensions_per_group[m]):
            mean=np.zeros([orginial_feature_dimension_size,1])
            for r in Best_samples_indices.reshape(-1):
                for t in range(0,Time):
                    Phi=Realization[r][0]['Basis'][:,t]
                    a=Realization[r][0]['Actions'][startDim + j,t]
                    reward=Realization[r][0]['Reward'][0]
                    tau_m_Erw=tau_temp[m][0]['A'] / tau_temp[m][0]['B']
                    Phi = Phi.reshape(-1,1)
                    rr2 = np.dot(np.dot(Phi,(a - np.dot(W_temp[m][0][j][0]['Mean'],Z_temp[r][0][t][0]['Mean'].T))),(tau_m_Erw)) / (np.dot(Phi.T,Phi))
                    mean=mean + rr2*reward[t]
                    
            mean=mean / sum_of_rewards
            mean=mean + (M[m][0][j].T / sigma2_M).reshape(-1,1)
            mean=np.dot(M_temp[m][0][j][0]['Cov'],mean)
            M_temp[m][0][j][0]['Mean'] = np.copy(mean.T)
       
    return M_temp
    

def update_W_Covariance(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,alpha_temp,tau_temp,latent_dimension_size,number_of_groups,Time,sum_of_rewards,dimensions_per_group):
        
        Sigma=np.zeros([latent_dimension_size,latent_dimension_size])
        
        for r in Best_samples_indices.reshape(-1):
            reward=Realization[r][0]['Reward'][0]
            Basis=Realization[r][0]['Basis']
            for t in range(0,Time):
                Sigma=Sigma + np.dot(reward[t],(np.dot(Z_temp[r][0][t][0]['Mean'].T,Z_temp[r][0][t][0]['Mean']) + Z_temp[r][0][t][0]['Cov'])) / (np.dot(Basis[:,t].T,Basis[:,t]))
        
        Sigma=Sigma / sum_of_rewards
        
        for m in range(0,number_of_groups):
            alpha_mk=np.zeros([latent_dimension_size,1])
            for k in range(0,latent_dimension_size):
                alpha_mk[k]=alpha_temp[m][0][k][0]['A'] / alpha_temp[m][0][k][0]['B']
            
            alpha_mk = diag(alpha_mk.reshape(-1))
            Sigma_tmp= inv(np.dot(Sigma,(tau_temp[m][0]['A'] / tau_temp[m][0]['B'])) + alpha_mk)
            
            for j in range(0,dimensions_per_group[m]):
                W_temp[m][0][j][0]['Cov'] = np.copy(Sigma_tmp)
       
        return W_temp

def update_W_Mean(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,tau_temp,latent_dimension_size,number_of_groups,Time,sum_of_rewards,dimensions_per_group):
    
    for m in range(0,number_of_groups):
        startDim = sum(dimensions_per_group[:m])

        for j in range(0,dimensions_per_group[m]):
            mean=np.zeros([latent_dimension_size,1])
            m_mj=M_temp[m][0][j][0]['Mean']
            for r in Best_samples_indices.reshape(-1):
                reward=Realization[r][0]['Reward'][0]
                action=Realization[r][0]['Actions']
                Basis=Realization[r][0]['Basis']
                tau_m_Erw=tau_temp[m][0]['A'] / tau_temp[m][0]['B']
                for t in range(0,Time):
                    mean=mean + (np.dot(np.dot(np.dot(reward[t],(action[startDim + j,t] - np.dot(m_mj,Basis[:,t].reshape(-1,1)))),Z_temp[r][0][t][0]['Mean']),(tau_m_Erw)) / (np.dot(Basis[:,t].T,Basis[:,t]))).T
            mean=mean / sum_of_rewards
            mean=np.dot(W_temp[m][0][j][0]['Cov'],mean)
            W_temp[m][0][j][0]['Mean'] = mean.T
    
    return W_temp


def update_Z_Mean_and_Covariance(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,tau_temp,orginial_feature_dimension_size,latent_dimension_size,number_of_groups,Time,dimensions_per_group):
        
    
        W_static=np.zeros([latent_dimension_size,latent_dimension_size])
        W_static_mean = np.empty([number_of_groups,1],dtype=object)
        M_static_mean= np.empty([number_of_groups,1],dtype=object)
        
        for m in range(0,number_of_groups):
            W_m_mean=np.zeros([dimensions_per_group[m],latent_dimension_size])
            M_m_mean=np.zeros([dimensions_per_group[m],orginial_feature_dimension_size])
            tau_tmp=tau_temp[m][0]['A'] / tau_temp[m][0]['B']
            for j in range(0,dimensions_per_group[m]):
                W_m_mean[j]=W_temp[m][0][j][0]['Mean']
                M_m_mean[j]=M_temp[m][0][j][0]['Mean']
            W_static_mean[m][0]=W_m_mean
            M_static_mean[m][0]=M_m_mean
            W_static=W_static + np.dot((np.dot(dimensions_per_group[m],W_temp[m][0][1][0]['Cov']) + np.dot(W_m_mean.T,W_m_mean)),tau_tmp)
        
        
        for r in Best_samples_indices.reshape(-1):
            Basis=Realization[r][0]['Basis']
            Action=Realization[r][0]['Actions']
            for t in range(0,Time):
                nenomm = np.dot(Basis[:,t].T,Basis[:,t])
                denomm = (np.eye(latent_dimension_size) + W_static)
                Z_temp[r][0][t][0]['Cov'] = np.copy(inv(inv(np.array(nenomm).reshape(1,-1))[0][0]*denomm))
                mean=np.zeros([latent_dimension_size,1])
                for m in range(0,number_of_groups):
                    #dimmm = np.array([0,4,6,10,12])
                    #dimmm = np.array([0,7])
                    aaa = Action[(sum(dimensions_per_group[:m])):(sum(dimensions_per_group[:m])+ dimensions_per_group[m] ),t]
                    aaa2 = ( aaa - np.dot(M_static_mean[m][0],Basis[:,t]))
                    aaa3 = (np.dot(W_static_mean[m][0].T,aaa2)) / (np.dot(Basis[:,t].T,Basis[:,t]))
                    mean=mean + np.dot(aaa3.reshape(-1,1),(tau_temp[m][0]['A'] / tau_temp[m][0]['B']))
                Z_temp[r][0][t][0]['Mean'] = np.copy(np.dot(Z_temp[r][0][t][0]['Cov'],mean))
                Z_temp[r][0][t][0]['Mean'] = Z_temp[r][0][t][0]['Mean'].T
                
        
        return Z_temp
    

def update_tau(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,tau_temp,orginial_feature_dimension_size,latent_dimension_size,number_of_groups,Time,sum_of_rewards,dimensions_per_group,tauB):
    
    
    for m in range(0,number_of_groups):
        M_tmp=np.zeros([dimensions_per_group[m],orginial_feature_dimension_size])
        W_tmp=np.zeros([dimensions_per_group[m],latent_dimension_size])
        for j in range(0,dimensions_per_group[m]):
            M_tmp[j]=M_temp[m][0][j][0]['Mean']
            W_tmp[j]=W_temp[m][0][j][0]['Mean']
        Bm=0
        for r in Best_samples_indices.reshape(-1):
            reward=Realization[r][0]['Reward'][0]
            basis=Realization[r][0]['Basis']
            actions=Realization[r][0]['Actions']
            startDim = sum(dimensions_per_group[:m])

            dims_start=startDim
            for t in range(0,Time):
                
                actions_t=actions[dims_start:(dims_start + dimensions_per_group[m]),t]
                Exp_Wt_W=np.dot(dimensions_per_group[m],W_temp[m][0][1][0]['Cov']) + np.dot(W_tmp.T,W_tmp)
                a1 = (np.dot(basis[:,t].T,basis[:,t]))
                a2 = (np.dot(actions_t.T,actions_t)   + np.dot(np.dot(np.dot(- 2,actions_t.T),M_tmp),basis[:,t]) + np.dot(np.dot(np.dot(np.dot(2,Z_temp[r][0][t][0]['Mean']),W_tmp.T),M_tmp),basis[:,t]) + np.dot(np.dot(np.dot(- 2,actions_t.T),W_tmp),Z_temp[r][0][t][0]['Mean'].T) + np.dot(np.dot(basis[:,t].T,(np.dot(dimensions_per_group[m],M_temp[m][0][1][0]['Cov']) + np.dot(M_tmp.T,M_tmp))),basis[:,t]) + np.trace(np.dot(Exp_Wt_W,Z_temp[r][0][t][0]['Cov'])) + np.dot(np.dot(Z_temp[r][0][t][0]['Mean'],Exp_Wt_W),Z_temp[r][0][t][0]['Mean'].T))
                Bm=Bm+ np.dot(reward[t]/a1,a2)
        
        Bm=Bm / sum_of_rewards
        Bm=Bm / 2
        tau_temp[m][0]['B'] = tauB + Bm[0][0]
        
    
    return tau_temp
    
def update_alpha(M_temp,M_old_temp,W_temp,alpha_temp,alphaB,latent_dimension_size,number_of_groups,dimensions_per_group):
    
    valueNorm_M_mean=0
    
    for m in range(0,number_of_groups):
        W_tmp=np.zeros([dimensions_per_group[m],latent_dimension_size])
        sum_variances=np.zeros([1,latent_dimension_size])
        for j in range(0,dimensions_per_group[m]):
            W_tmp[j]=W_temp[m][0][j][0]['Mean']
            valueNorm_M_mean=valueNorm_M_mean + norm(M_old_temp[m][0][j][0]['Mean'] - M_temp[m][0][j][0]['Mean'],2)
            
            for k in range(0,latent_dimension_size):
                sum_variances[0][k]=sum_variances[0][k] + W_temp[m][0][j][0]['Cov'][k,k]
        for k in range(0,latent_dimension_size):
            alpha_temp[m][0][k][0]['B'] = np.copy(alphaB + np.dot(W_tmp[:,k].T,W_tmp[:,k]) / 2 + sum_variances[0][k] / 2)
    
    return alpha_temp,valueNorm_M_mean