import pdb
from oct2py import octave
from numpy import *
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import scipy.stats
import time
import math
from datetime import datetime
import socket
    

def get_samples(W=None,M=None,tau=None,Latent=None,DimPerGroup=None,Time=None,rendering=0,*args,**kwargs):

    DoF=sum(DimPerGroup)
    number_of_groups=DimPerGroup.shape[0]    
    
    means = np.arange(-3,Time+3,3)
    BasisDim = len(means)
    variance = 3
    bafu = scipy.stats.norm.pdf(1,means,variance)
    bafu = bafu.reshape(-1,1)
    
    for i in range(1,Time):
        function_i = scipy.stats.norm.pdf(i+1,means,variance)
        function_i = function_i.reshape(-1,1)
        bafu = np.concatenate((bafu,function_i),1)
    
    
    sum_bafu = np.sum(bafu,0)
    sum_bafu = sum_bafu.reshape(1,-1)
    sum_bafu = np.repeat(sum_bafu,len(means),0)
    Basisfunctions = np.divide(bafu,sum_bafu)
    
    Z=octave.randn(Latent,BasisDim)
    Actions=np.zeros([DoF,Time])
    
    reward=np.zeros([1,Time])
     
    for t in arange(0,Time).reshape(-1):
        for m in arange(0,number_of_groups).reshape(-1):
            
            startDim = sum(DimPerGroup[:m])
            xx = octave.eval('normrnd(0,'+str(octave.inv(tau[m][0]) + 2)+','+str(DimPerGroup[m])+','+str(BasisDim)+');')
            
            Actions[startDim:(startDim + DimPerGroup[m]),t]=dot(dot(W[m][0],Z),Basisfunctions[:,t]) + dot(M[m][0],Basisfunctions[:,t]) + dot(xx,Basisfunctions[:,t])
        
        
        CurrentAngle=Actions[:,t]
        

        try:
            ssr = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
            port = 55001               
            #ssr.connect(('192.168.125.1', port))
            ssr.connect(('127.0.0.1', port))

            ssr2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
            port2 = 44000               
            #ssr2.connect(('192.168.125.1', port2))
            ssr2.connect(('127.0.0.1', port2))
            
            # receive data from the server
            #print(ssr.recv(1024))
            #print(ssr2.recv(1024))
            ssr.recv(1024)
            ssr2.recv(1024)

            # send data to the server
            writing_main_data = [["-110.08","-128.668","51.0821","-3.07672","56.8406","-159.638","50.8102"]]
            original_angles = ["-111.254","-134.357","-0.0150254","1.35811","110.899","-164.631","47.7262"]
            
            for i in range(len(writing_main_data)):
                writing_data = writing_main_data[i]            
                for i in range(len(writing_data)):
                    #message = float(original_angles[i]) + CurrentAngle[i]/100
                    message = CurrentAngle[i]/25
                    message = str(message)
                    message = message.encode('utf-8')

                    message2 = CurrentAngle[i+7]/25
                    message2 = str(message2)
                    message2 = message2.encode('utf-8')

                    #pdb.set_trace()
                    ssr.send(message)
                    #print(ssr.recv(1024))
                    ssr.recv(1024)

                    ssr2.send(message2)
                    #print(ssr2.recv(1024))
                    ssr2.recv(1024)

                received_point = ssr.recv(1024).decode('utf-8')
                y_received = float(received_point.strip('[]').split(',')[1])
                height_received = float(received_point.strip('[]').split(',')[2])
                #print(height_received,t)

                received_point2 = ssr2.recv(1024).decode('utf-8')
                y_received2 = float(received_point2.strip('[]').split(',')[1])
                height_received2 = float(received_point2.strip('[]').split(',')[2])
                #print(height_received2,t)


            if(t == Time -1):
                ss = "3333"
                ssr.send(ss.encode('utf-8'))
                ssr.recv(1024)

                ssr2.send(ss.encode('utf-8'))
                ssr2.recv(1024)
            else:
                ss = "9999"
                ssr.send(ss.encode('utf-8'))
                ssr.recv(1024)

                ssr2.send(ss.encode('utf-8'))
                ssr2.recv(1024)
        
        except:
            print("You closed")
            pdb.set_trace()

        expected_position = [0,142,450]
        actual_position = [0,y_received,height_received]
        expected_position = np.array(expected_position) 
        actual_position = np.array(actual_position)

        expected_position2 = [0,-142,450]
        actual_position2 = [0,y_received2,height_received2]
        expected_position2 = np.array(expected_position2)
        actual_position2 = np.array(actual_position2)

        reward1 = max(norm(actual_position - expected_position,2),0)
        reward2 = max(norm(actual_position2 - expected_position2,2),0)
        reward[0][t] = reward1 + reward2
    '''    
    for i in range(Time): 
        reward[0][i]=max(norm(actual_position - expected_position,2),0)
    '''
        
    reward2=exp(- reward)
    reward = dot(np.ones(reward2.shape),sum(reward2))
    Z=dot(Z,Basisfunctions)

    return Actions,Basisfunctions,reward,Z
    
if __name__ == '__main__':
    pass
    
