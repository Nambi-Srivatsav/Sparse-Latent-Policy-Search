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
    
def GrouPS(*args,**kwargs):
    
    
    best_size=10
    sample_size=20
    number_of_groups=2
    dimensions_per_group=np.array([7,7])
    latent_dimension_size=6
    orginial_feature_dimension_size=6
    Time=10
    max_iterations=100
    max_inner_iterations=20
    sigma2_M=100
    anti_convergence_factor=1.5
    tauA=1000
    tauB=1000
    alphaA=1
    alphaB=1
    initial_Tau= inv(np.array([10]).reshape(1,-1)**1.5)[0][0]
    initial_Alpha=1
    
    Iterations_number=0
    Inner_iterations_number=0
    reward_plot= np.zeros([1,max_iterations])
    check_if_converged=True
    check_if_converged2=True
    
    W=np.empty([number_of_groups,1],dtype=object)
    tau = np.empty([number_of_groups,1],dtype=object)
    alpha=np.empty([number_of_groups,1],dtype=object)
    initialM = np.empty([number_of_groups,1],dtype=object)
    
    for m in range(0,number_of_groups):
        W[m][0] = np.random.randn(dimensions_per_group[m],latent_dimension_size)
        tau[m][0]=initial_Tau
        alpha[m]=initial_Alpha

    
    for m in range(0,number_of_groups):
        initialM[m][0]=np.zeros([dimensions_per_group[m],orginial_feature_dimension_size])
    
    
    
    M = np.copy(initialM)
    
    '''
    M = np.load('M.npy')
    W = np.load('W.npy')
    tau = np.load('tau.npy')
    alpha = np.load('alpha.npy')
    '''

    get_samples(W,M,tau,latent_dimension_size,dimensions_per_group,Time,rendering=1,nout=4)
    
    while check_if_converged:
        check_if_converged2=True
        sum_of_rewards=0
        
        ###################  Collect multiple samples with existing parameters ###############
        
        if sample_size==sample_size or Iterations_number == 0:
            Realization=np.empty([sample_size,1],dtype=object)
            Z_temp=np.empty([sample_size,1],dtype=object)
            for r in range(0,sample_size):
                Actions,Basisfunctions,Reward,Zvals= get_samples(W,M,tau,latent_dimension_size,dimensions_per_group,Time,nout=4)
                Realization[r][0] = {'Actions':Actions, 'Basis':Basisfunctions,'Reward':Reward}
                Z_temp[r][0]=np.empty([Time,1],dtype=object)
                
                for t in range(0,Time):
                    Z_temp[r][0][t][0] = {'Mean':Zvals[:,t],'Cov':np.eye(latent_dimension_size)}
                    Z_temp[r][0][t][0]['Mean'] = Z_temp[r][0][t][0]['Mean'].reshape(1,-1) 
                    
        reward_tmp = np.empty([sample_size,1],dtype=object)
        
        for r in range(0,sample_size):
            reward_tmp[r][0]=np.sum(Realization[r][0]['Reward'])
        
        Idx=[i[0] for i in sorted(enumerate(reward_tmp), key=lambda x:-x[1])]
        Best_samples_indices=np.array(Idx[0:best_size])
        sum_of_rewards=0
        for r in Best_samples_indices.reshape(-1):
            sum_of_rewards=sum_of_rewards + np.sum(Realization[r][0]['Reward'])
        
        
        ################## Initialize various distributions #######################
        
        W_temp = np.empty([number_of_groups,1],dtype=object)
        M_temp= np.empty([number_of_groups,1],dtype=object)
        tau_temp=np.empty([number_of_groups,1],dtype=object)
        alpha_temp=np.empty([number_of_groups,1],dtype=object)
        
        for m in range(0,number_of_groups):
            W_temp[m][0]=np.empty([dimensions_per_group[m],1],dtype=object)
            M_temp[m][0]=np.empty([dimensions_per_group[m],1],dtype=object)
            alpha_temp[m][0]= np.empty([latent_dimension_size,1],dtype=object)
            for j in range(0,dimensions_per_group[m]):
                W_temp[m][0][j][0] = {'Mean':W[m][0][j,:],'Cov':np.eye(latent_dimension_size)}
                W_temp[m][0][j][0]['Mean'] = W_temp[m][0][j][0]['Mean'].reshape(1,-1)
                
                M_temp[m][0][j][0] = {'Mean':M[m][0][j,:],'Cov':np.eye(latent_dimension_size)}
                M_temp[m][0][j][0]['Mean'] = M_temp[m][0][j][0]['Mean'].reshape(1,-1)
                
            tau_temp[m][0] = {'A':tauA + np.dot(np.dot(0.5,dimensions_per_group[m]),Time),'B':tauB + np.dot(np.dot(0.5,dimensions_per_group[m]),Time)}
            for k in range(0,latent_dimension_size):
                alpha_temp[m][0][k][0]={'A':alphaA + dimensions_per_group[m] / 2,'B':alphaB + dimensions_per_group[m] / 2}
        
        Inner_iterations_number=0
        
        
        
        ############################### Start the iterations #######################
        
        while check_if_converged2:
            
            M_old_temp = np.empty([number_of_groups,1],dtype=object)
            
            for m in range(0,number_of_groups): 
                M_old_temp[m][0]=np.empty([dimensions_per_group[m],1],dtype=object)
            
                for j in range(0,dimensions_per_group[m]):
                    M_old_temp[m][0][j] = {'Mean':np.copy(M_temp[m][0][j][0]['Mean']),'Cov':np.copy(M_temp[m][0][j][0]['Cov'])}
                    M_old_temp[m][0][j][0]['Mean'] = M_old_temp[m][0][j][0]['Mean'].reshape(1,-1)
                    
            
            
            ####################### update M - Covariance ##############################
            
           
            M_temp = update_M_Covariance(Best_samples_indices,Realization,M_temp,tau_temp,orginial_feature_dimension_size,number_of_groups,sigma2_M,Time,sum_of_rewards,dimensions_per_group)
            
            ####################### update M - Mean ###################################
            
            
            M_temp = update_M_Mean(Best_samples_indices,Realization,M_temp,M,W_temp,Z_temp,tau_temp,orginial_feature_dimension_size,number_of_groups,sigma2_M,Time,sum_of_rewards,dimensions_per_group)
            
            ####################### update W - Covariance ##############################
            
            
            W_temp = update_W_Covariance(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,alpha_temp,tau_temp,latent_dimension_size,number_of_groups,Time,sum_of_rewards,dimensions_per_group)

            ####################### update W - Mean ################################
            
                    
            W_temp = update_W_Mean(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,tau_temp,latent_dimension_size,number_of_groups,Time,sum_of_rewards,dimensions_per_group)
            
            ####################### update Z - Mean and Covariance ##############################
                    
            
            Z_temp = update_Z_Mean_and_Covariance(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,tau_temp,orginial_feature_dimension_size,latent_dimension_size,number_of_groups,Time,dimensions_per_group)

            ####################### update Tau B ##############################
            
            tau_temp = update_tau(Best_samples_indices,Realization,M_temp,W_temp,Z_temp,tau_temp,orginial_feature_dimension_size,latent_dimension_size,number_of_groups,Time,sum_of_rewards,dimensions_per_group,tauB)
            ####################### update alpha B ##############################
            
            alpha_temp,valueNorm_M_mean = update_alpha(M_temp,M_old_temp,W_temp,alpha_temp,alphaB,latent_dimension_size,number_of_groups,dimensions_per_group)
            
            check_if_converged2=valueNorm_M_mean > 0.01
            print(Iterations_number,Inner_iterations_number,valueNorm_M_mean)
            
            if Inner_iterations_number >= max_inner_iterations:
                check_if_converged2=0
            else:
                Inner_iterations_number=Inner_iterations_number + 1
        
        
        
        ############################# Copy the learnt values of M,W,alpha and tau ###################
        
       
        for xx in range(0,number_of_groups): 
                alpha[xx][0] = np.empty([latent_dimension_size,1],dtype=object)
                
        for m in range(0,number_of_groups):
            for j in range(0,dimensions_per_group[m]):
                M[m][0][j]=M_temp[m][0][j][0]['Mean'][0]
                W[m][0][j]=np.dot(anti_convergence_factor,W_temp[m][0][j][0]['Mean'])[0]
                
            tau[m][0]= lin.solve(np.array([anti_convergence_factor]).reshape(1,-1),np.array((tau_temp[m][0]['A'] / tau_temp[m][0]['B'])).reshape(1,-1))[0][0]
            
            for k in range(0,latent_dimension_size): 
                alpha[m][0][k]=alpha_temp[m][0][k][0]['A'] / alpha_temp[m][0][k][0]['B']
                        
        
        Iterations_number=Iterations_number + 1
        check_if_converged=Iterations_number < max_iterations
        
        
        np.save('M.npy',M)
        np.save('W.npy',W)
        np.save('tau.npy',tau)
        np.save('alpha.npy',alpha)
        np.save('rewardOverTime.npy',M)
        
        if(Iterations_number%5 == 0):   
            os.mkdir(str(Iterations_number))
            np.save('./'+str(Iterations_number)+'/M.npy',M)
            np.save('./'+str(Iterations_number)+'/W.npy',W)
            np.save('./'+str(Iterations_number)+'/tau.npy',tau)
            np.save('./'+str(Iterations_number)+'/alpha.npy',alpha)
            np.save('./'+str(Iterations_number)+'/rewardOverTime.npy',M)
            
        get_samples(W,M,tau,latent_dimension_size,dimensions_per_group,Time,rendering=1,nout=4)

        print('Iteration :',Iterations_number)
    
    rewardOverTime=np.copy(reward_plot)
    return M,W,tau,alpha,rewardOverTime
    
if __name__ == '__main__':
    
    octave.eval('randn("seed", 10)', verbose=True)
    np.random.seed(10)
    M,W,tau,alpha,rewardOverTime=GrouPS()
    '''
    np.save('M.npy',M)
    np.save('W.npy',W)
    np.save('tau.npy',tau)
    np.save('alpha.npy',alpha)
    np.save('rewardOverTime.npy',M)
    '''