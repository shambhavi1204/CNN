import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
data_train = pd.read_csv(r"C:\Users\sinha\Downloads\train.csv")
data_test = pd.read_csv(r"C:\Users\sinha\Downloads\test.csv")

data = np.array(data_train)
datat = np.array(data_test)

Y_train=data[:,0]
X_train=data[:,1:785]

Y_test=datat[:,0]
X_test=datat[:,1:785]


filter_size = 3
num_filter = 5
output_size = 10
input_dim = 28
input_size = 28*28
k = np.random.randn(filter_size,filter_size,num_filter)/np.sqrt(filter_size)
#print(X)
W = np.random.randn(output_size,input_dim-filter_size+1,input_dim-filter_size+1,num_filter)/np.sqrt(input_dim-filter_size+1)
bias= np.zeros((output_size, 1))/np.sqrt(output_size)
    


LR = 0.0001
num_epochs = 10

for epochs in range(num_epochs):
    print(epochs)
    total_correct = 0
    for n in range(len(X_train)):
        if n%1000==0:
            print(n)
        
        x=X_train[n-1][:]
        y=Y_train[n-1]
        x=np.reshape(x,(input_dim, input_dim))
        Z=np.zeros([26,26,5])
        
        for i in range(num_filter):
            k1=np.rot90(k[:,:,i-1])
            k1=np.rot90(k1)
            Zq=ndimage.convolve(x, k1, mode='constant')
            Zq=Zq[1:27,1:27]
            Z[:,:,1]=Zq
            
        H=np.maximum(Z,0)
        U=np.zeros((output_size,1))
        
        for i in range(output_size):
            temp1=W[i,:,:,:]
            temp2=np.multiply(temp1,H)
            U[i]=np.sum(temp2) + bias[i]

        rho=np.exp(U - max(U))/np.sum(np.exp(U - max(U)))
        predicted_value = np.argmax(rho)

        if (predicted_value == y):
            total_correct += 1

        arr = np.zeros((output_size,1))
        arr[y]=1
        diff_U=rho-arr
        diff_bias=diff_U
        diff_W=np.zeros((output_size,input_dim-filter_size+1,input_dim-filter_size+1,num_filter))

        for i in range(output_size):
        	diff_W[i,:,:,:]=diff_U[i]*H

        delta=np.zeros(H.shape)
        for i in range(input_dim-filter_size+1):
            for j in range(input_dim-filter_size+1):
                for p in range(num_filter):
                    delta[i,j,p]=np.sum(np.multiply(diff_U,W[:,i,j,p]))
                    

            grad_Zdel = np.multiply(np.where(Z>0,1,0),delta)
            #print(grad_Zdel.shape)
            diff_K=np.zeros([3,3,5])
            for j in range(num_filter):
                g1=np.rot90(grad_Zdel[:,:,j-1])
                g1=np.rot90(g1)
                dfk=ndimage.convolve(x, g1, mode='constant')
                dfk=dfk[0:3,0:3]
                diff_K[:,:,j-1]=dfk



            bias=bias - LR*diff_bias
            W = W - LR*diff_W
            k = k - LR*diff_K
    print("Training accuracy for epoch {} : {}".format(epochs+1, total_correct/np.float(len(x_train))))


total_correct = 0

for n in range(len(x_test)):
    y = Y_test[n]
    x = X_test[n][:]
    x = np.reshape(x, (input_dim, input_dim))

    Z=np.zeros([26,26,5])
        
    for i in range(num_filter):
        k1=np.rot90(k[:,:,i-1])
        k1=np.rot90(k1)
        Zq=ndimage.convolve(x, k[:,:,i-1], mode='constant')
        Zq=Zq[1:27,1:27]
        Z[:,:,1]=Zq

    H=np.maximum(Z,0)

    for i in range(output_size):
        temp1=W[i,:,:,:]
        temp2=np.multiply(temp1,H)
        U[i]=np.sum(temp2)+bias[i]
       	

    rho = np.exp(U - max(U))/np.sum(np.exp(U - max(U)))
    predicted_value = np.argmax(rho)

    if (predicted_value == y):
        total_correct += 1

print("Test accuracy : {}".format(total_correct/np.float(len(x_test))))
    
    
                
     		


        
    



