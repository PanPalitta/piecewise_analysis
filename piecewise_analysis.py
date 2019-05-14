# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:13:46 2017

@author: ppalitta

At a Glance:
This Python script performs regression analysis of a data set that is suspected 
of having a piecewise linear structure. The goal is to provide the user with
information regarding how well the data fit with each model to aid in model 
selection.

Feature:
    -Automatic segmentation
    -Fit to five piecewise models
        --single linear line
        --two-segment linear line
        --three-segment linear line
        --interpolation+two-segment linear line (3 segments in total)
        --interpolation+single linear line (2 segments in total)
    -Report slope, intercept and their corresponding uncertainty for each 
    segment.
    -Report adjusted R-squared, adjusted AIC, F-value and Mallows' Cp for each 
    model.
    -Produce fitted plots with data for visual inspection
    
Potential pitfalls:
    -Except for the three-segment linear fit, segmentation algorithm is greedy and has low tolerant to noise in the data.
    Smarter algorithm may be desirable in some cases.    

Installation:
The script uses piecewise regression library `pwlf.py' by Charles Jekel for 3-segment 
regression. To install this library, follow the instruction on Charles' blog 
[https://jekel.me/2017/Fit-a-piecewise-linear-function-to-data/] 
and copy the 'pwlf.py' file to the same folder as `piecewise_analysis.py'.

Input assumption in this version:
    - y data is assumed to be a column of double-precision floating-point 
    numerics.
    - x data is generated within the code.

"""

import matplotlib.pyplot as plt #For plotting the data and the fit
from scipy.interpolate import interp1d #For interpolation
from scipy import stats #For linear regression
import numpy as np

import pwlf #importing piecewise regression library

"""
Defining functions
"""

def even_breaks(x,NumSeg):
    array=np.zeros(NumSeg+1);
    for i in range(0,NumSeg+1):
        step=int(len(x)/NumSeg);
        if i==NumSeg:
            array[i]=x[-1];
        else:
            array[i]=x[i*step];
    return array

def seperateData(x,y,breaks):
    sepDataX = [[] for i in range(numseg)];
    sepDataY = [[] for i in range(numseg)];
    for i in range(0, numseg):
        dataX = []
        dataY = []
        aTest = x >= breaks[i]
        dataX = np.extract(aTest, x)
        dataY = np.extract(aTest, y)
        bTest = dataX <= breaks[i+1]
        dataX = np.extract(bTest, dataX)
        dataY = np.extract(bTest, dataY)
        sepDataX[i] = np.array(dataX)
        sepDataY[i] = np.array(dataY)
    return (sepDataX,sepDataY)
    
def piecewise_fit(x,y,numseg,breaks):
    #This function performs piecewise linear fit. This is the same algorithm
    #use in pwlf.py, except it can be used for any number of segments.
    numparam=numseg+1;
    sepDataX = [[] for i in range(numseg)];
    sepDataY = [[] for i in range(numseg)];
    sepDataX,sepDataY=seperateData(x,y,breaks);
    
    A = np.zeros([numparam, numparam])
    B = np.zeros(numparam)
    for i in range(0,numparam):
        if i != 0:
            #   first sum
            A[i,i-1] = A[i,i-1] - sum((sepDataX[i-1] - breaks[i-1]) * (sepDataX[i-1] - breaks[i])) / ((breaks[i] - breaks[i-1]) ** 2)
            A[i,i] = A[i,i] + sum((sepDataX[i-1] - breaks[i-1]) ** 2) / ((breaks[i] - breaks[i-1]) ** 2)
            B[i] = B[i] + (sum(sepDataX[i-1] * sepDataY[i-1]) - breaks[i-1] * sum(sepDataY[i-1])) / (breaks[i] - breaks[i-1])
        
        if i != numparam - 1:
                    #   second sum
            A[i,i] = A[i,i] + sum(((sepDataX[i] - breaks[i+1]) ** 2)) / ((breaks[i+1] - breaks[i]) ** 2)
            A[i,i+1] = A[i,i+1] - sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i+1])) / ((breaks[i+1] - breaks[i]) ** 2)
            B[i] = B[i] + (-sum(sepDataX[i] * sepDataY[i]) + breaks[i+1] * sum(sepDataY[i])) / (breaks[i+1] - breaks[i])
        
    p=np.linalg.solve(A,B)
    
    yHat = []
    yHat_temp=[]
    
    yHat_len=0;        
    for i,j in enumerate(sepDataX):
        m = (p[i+1] - p[i])/(breaks[i+1]-breaks[i])
        yHat_temp.append(m*(j-breaks[i]) + p[i])
        yHat_len=yHat_len+len(yHat_temp[i]);
    
    yHat=yHat_temp[0];
    if yHat_len>len(x): #This is the case where a data point overlap between segments
        for i in range(1,numseg):
            temp=np.zeros(len(sepDataX[i])-1)
            for j in range(1, len(yHat_temp[i])):
                temp[j-1]=yHat_temp[i][j]
            yHat = np.concatenate((yHat,temp))
    else:
        yHat = np.concatenate(yHat_temp)    
    
    return (p, yHat)
    
def InterPiecewiseLinearModel(x,y,numseg,breaks):
    #This function is used to fit piecewise linear model where the first segment
    #is an interpolation. This can beused for more than 3 segments.
    sepDataX = [[] for i in range(numseg)];
    sepDataY = [[] for i in range(numseg)];
    sepDataX,sepDataY=seperateData(x,y,breaks);

    yHat = []

    #first segment
    f = interp1d(sepDataX[0],sepDataY[0])
    yHat=f(sepDataX[0])

    #second and third segment
    numpiece=numseg-1;
    numparam=numpiece+1;

    A = np.zeros([numparam, numparam])
    B = np.zeros(numparam)

    for i in range(0,numparam):
        if i != 0:
            #   first sum
            A[i,i-1] = A[i,i-1] - sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i+1])) / ((breaks[i+1] - breaks[i]) ** 2)
            A[i,i] = A[i,i] + sum((sepDataX[i] - breaks[i]) ** 2) / ((breaks[i+1] - breaks[i]) ** 2)
            B[i] = B[i] + (sum(sepDataX[i] * sepDataY[i]) - breaks[i] * sum(sepDataY[i])) / (breaks[i+1] - breaks[i])
        
        if i != numparam - 1:
            #   second sum
            A[i,i] = A[i,i] + sum(((sepDataX[i+1] - breaks[i+2]) ** 2)) / ((breaks[i+2] - breaks[i+1]) ** 2)
            A[i,i+1] = A[i,i+1] - sum((sepDataX[i+1] - breaks[i+1]) * (sepDataX[i+1] - breaks[i+2])) / ((breaks[i+2] - breaks[i+1]) ** 2)
            B[i] = B[i] + (-sum(sepDataX[i+1] * sepDataY[i+1]) + breaks[i+2] * sum(sepDataY[i+1])) / (breaks[i+2] - breaks[i+1])

    p=np.linalg.solve(A,B)

    lenSepX=0;
    for i in range(numseg):
        lenSepX=lenSepX+len(sepDataX[i])

    if lenSepX>len(x):
        yHat_temp=[]
        for i in range(1,numseg):
            m = (p[i] - p[i-1])/(breaks[i+1]-breaks[i])
            for j in range(len(sepDataX[i])-1):
                yHat_temp.append(m*(sepDataX[i][j+1]-breaks[i]) + p[i-1])
        yHat = np.concatenate((yHat,yHat_temp));
    else:
        for i in range(1,numseg):
            m = (p[i] - p[i-1])/(breaks[i+1]-breaks[i])
            for j in range(len(sepDataX[i])):
                yHat_temp.append(m*(sepDataX[i][j]-breaks[i]) + p[i-1])
        yHat = np.concatenate((yHat,yHat_temp));    
    return (p, yHat)

def InterLinear(x,y,breaks):
    #This function fits 2-segment model where the first segment is an interpolation.
    numseg = 2;

    sepDataX = [[] for i in range(numseg)];
    sepDataY = [[] for i in range(numseg)];
    sepDataX,sepDataY=seperateData(x,y,breaks);

    yHat = []

    #first segment
    f = interp1d(sepDataX[0],sepDataY[0])
    yHat=f(sepDataX[0])

    #second segment
    slope, intercept, r_value, p_value, std_err=stats.linregress(sepDataX[1],sepDataY[1]);

    lenSepX=0;
    for i in range(numseg):
        lenSepX=lenSepX+len(sepDataX[i])
    
    if lenSepX>len(x):
        yHat_temp=[]
        for i in range(1,numseg):
            for j in range(len(sepDataX[i])-1):
                yHat_temp.append(slope*(sepDataX[i][j+1]) + intercept)
        yHat = np.concatenate((yHat,yHat_temp));
    else:
        for i in range(1,numseg):
            for j in range(len(sepDataX[i])):
                yHat_temp.append(slope*(sepDataX[i][j]) + intercept)
        yHat = np.concatenate((yHat,yHat_temp));    

    return (slope,intercept,yHat)
    
def res_squared(y,yHat):
    #This function calculate the difference between two arrays.
    error=y-yHat;
    res=np.dot(error.T,error)
    return res

def ThreePieceModel(x,y):
    #This function specifically fit 3-piece model where the first segment is an
    #interpolation.
    numseg = 3;
    step=int(len(x)/numseg);

    breaks=np.zeros(numseg+1);
    res=np.zeros(step-1)
    index=np.zeros(step-1)
    
    for i in range(1,step):
        breaks=[x[0],x[i],x[len(x)-2],x[-1]]
        p, yHat=InterPiecewiseLinearModel(x,y,numseg,breaks)
        res[i-1]=res_squared(y,yHat)
        index[i-1]=i;

    #The breaking point is chosen such that the derivative of the residual is 
    #minimized. The residual itself cannot be used, otherwise the plot will be
    #all interpolation.
    diff_res=np.gradient(res);
    stop_point=np.argmin(diff_res)+1      
    if np.argmin(res)!=len(res)-1:
        stop_point=np.argmin(res)+1
    
    #The lines below plot the residual and its derivative to check that the 
    #segmentation to verify that our condition is giving satisfying result.
    #The plots can be turned off without negative impact to the fitting.     
    plt.plot(res)
    plt.show()
    plt.plot(diff_res)
    plt.show()
    print(stop_point)

    breaks[1]=x[stop_point];
    
    step=int(len(x)-stop_point-1);
    res=np.zeros(step-1)
    index=np.zeros(step-1)
    for i in range(1,step):
        breaks[2]=x[stop_point+i];
        p, yHat=InterPiecewiseLinearModel(x,y,numseg,breaks)
        res[i-1]=res_squared(y,yHat)
        index[i-1]=i;

    breaks[2]=x[np.argmin(res)+1+stop_point];
    p, yHat=InterPiecewiseLinearModel(x,y,numseg,breaks)   
    return (p,yHat,breaks)    

def TwoPiecewiseLinearModel(x,y):
    numseg =2;
    step=2*int(len(x)/numseg);
    
    breaks=np.zeros(numseg+1);
    res=np.zeros(step-1)
    index=np.zeros(step-1)

    for i in range (1,step):
        breaks=[x[0],x[i],x[-1]];
        p, yHat = piecewise_fit(x,y,numseg,breaks)
        res[i-1]=res_squared(y,yHat)
        index[i-1]=i;    

    diff_res=np.gradient(res);
    stop_point=np.argmin(diff_res)+1
    if np.argmin(res)!=len(res)-1:
        stop_point=np.argmin(res)+1

    #The lines below plot the residual and its derivative to check that the 
    #segmentation to verify that our condition is giving satisfying result.
    #The plots can be turned off without negative impact to the fitting.             
    plt.plot(res)
    plt.show()
    plt.plot(diff_res)
    plt.show()
    
    breaks[1]=x[stop_point];
    p, yHat = piecewise_fit(x,y,numseg,breaks)
    return (p,yHat,breaks)
    
def TwoPieceInterModel(x,y):
    numseg=2;
    step=int(len(x)/numseg);

    breaks=np.zeros(numseg+1);
    res=np.zeros(step-1)
    index=np.zeros(step-1)   
    
    for i in range(1,step):
        breaks=[x[0],x[i],x[-1]]    
        slope,intercept,yHat=InterLinear(x,y,breaks)
        res[i-1]=res_squared(y,yHat)
        index[i-1]=i;

    #The breaking point is chosen such that the derivative of the residual is 
    #minimized. The residual itself cannot be used, otherwise the plot will be
    #all interpolation.
    diff_res=np.gradient(res);
    stop_point=np.argmin(diff_res)+1
    if np.argmin(res)!=len(res)-1:
        stop_point=np.argmin(res)+1
    
    #The lines below plot the residual and its derivative to check that the 
    #segmentation to verify that our condition is giving satisfying result.
    #The plots can be turned off without negative impact to the fitting.       
    plt.plot(res)
    plt.show()
    plt.plot(diff_res)
    plt.show()
    print(stop_point)

    breaks[1]=x[stop_point];
    slope,intercept,yHat=InterLinear(x,y,breaks)
    return (slope,intercept,yHat,breaks)    

"""
###############The script for fitting the data starts here.###################
"""


"""
Setting up
"""

#Loading data
#The script assumes that the data is a column of double-precision floating-point numerics.

#Y array upload
result_file='example/Gaussian30.txt';#put the location of the file here.
y=np.loadtxt(result_file);

#X array generation
x=np.zeros(len(y));
for i in range(len(y)):
    x[i]=np.log10(4+i);

#Create output file. File is refreshed everytime the script is run.
output=open("example/FitResult.txt","w") #put the location of the output here.
output.write("|Model|\t m \t b\t R^2_adjusted \t AIC_adjusted \t F \t Cp \t break points\n")
#This is the headinging of the output.

numdata=len(y)
y_mean=sum(y)/numdata;

"""
###Three-piecewise linear regression (assumed full model)
"""
#Basic information about the model
numseg=3;
numparam=6;

#Regression
myPWLF = pwlf.piecewise_lin_fit(x,y)
res = myPWLF.fit(3, disp=True)
yHat = myPWLF.predict(x)

p_full=myPWLF.fitParameters
breaks_full=myPWLF.fitBreaks

#Compute model parameters and uncertainties
m=[(p_full[1]-p_full[0])/(breaks_full[1]-breaks_full[0]),(p_full[2]-p_full[1])/(breaks_full[2]-breaks_full[1]),(p_full[3]-p_full[2])/(breaks_full[3]-breaks_full[2])]
b=[p_full[0],p_full[1],p_full[2]]

#Regression Analysis
#Residual computation
Sm=[0,0,0];
Sb=[0,0,0];
sepDataX, sepDataY=seperateData(x,y,breaks_full)
sepDataX, sepHatY=seperateData(x,yHat,breaks_full)
for i in range(numseg):
    num=len(sepDataX[i])
    x_mean=sum(sepDataX[i])/len(sepDataX[i])
    Sy=sum((sepDataY[i]-sepHatY[i])**2)
    Sx=sum((sepDataX[i]-x_mean)**2)
    SS=sum(sepDataX[i]**2)
    Sm[i]=Sy/(Sx*(num-2))
    Sb[i]=Sy*SS/(Sx*num*(num-2))
SSR_full=res_squared(y,yHat)

#R squared
R=1-SSR_full/sum((y-y_mean)**2)
Rc_full=R-(numparam*(1-R)/(numdata-numparam-1));#adjusted R-squared

#AIC
V_full=SSR_full/(numdata-numparam-1)
AIC=numdata*np.log(V_full)+numdata*np.log(float(numdata-numparam-1)/numdata)+2*numdata+4;
AIC_full=AIC+(2*(numparam+2)*(numparam+3))/(numdata-numparam-3);#adjusted AIC for the full model

#Measure for comparison between models
#can't do F for this model since this is the full model. Cp is used as baseline.
Cp=SSR_full/V_full-numdata+2*numparam

output.writelines('|3-piecewise linear|\t'+str(m)+'+/-'+str(Sm)+'\t'+str(b)+'+/-'+str(Sb)+'\t'+ str(Rc_full)+'\t'+str(AIC_full)+ '\t - \t'+str(Cp)+'\t'+str(breaks_full)+'\n')
#print('|3-piecewise linear|',m,'+/-',Sm,'|',b,'+/-',Sb,'|', Rc_full, AIC_full, ' - ',Cp,breaks_full)
plt.plot(x,y,'o',x,yHat,'-')
plt.show();

"""
###3 pieces with interpolation
"""
#Basic information about the model
numseg=3;
numparam=5;
constraints=6-numparam;
#Regression
p_full, yHat_full, breaks_full=ThreePieceModel(x,y)
#Compute model parameters and uncertainties
m=[(p_full[1]-p_full[0])/(breaks_full[2]-breaks_full[1]),(p_full[2]-p_full[1])/(breaks_full[3]-breaks_full[2])]
b=[p_full[0],p_full[1]]

#Regression Analysis
#Residual computation
Sm=[0,0];
Sb=[0,0];
sepDataX, sepDataY=seperateData(x,y,breaks_full)
sepDataX, sepHatY=seperateData(x,yHat_full,breaks_full)
for i in range(numseg-1):
    num=len(sepDataX[i+1])
    x_mean=sum(sepDataX[i+1])/len(sepDataX[i+1])
    Sy=sum((sepDataY[i+1]-sepHatY[i+1])**2)
    Sx=sum((sepDataX[i+1]-x_mean)**2)
    SS=sum(sepDataX[i+1]**2)
    Sm[i]=Sy/(Sx*(num-2))
    Sb[i]=Sy*SS/(Sx*num*(num-2))
SSR_ss=res_squared(y,yHat_full)

#R squared
R=1-SSR_ss/sum((y-y_mean)**2)
Rc_ss=R-(numparam*(1-R)/(numdata-numparam-1)); #adjusted R squared

#AIC
V_ss=SSR_ss/(numdata-numparam-1)
AIC=numdata*np.log(V_full)+numdata*np.log(float(numdata-numparam-1)/numdata)+2*numdata+4;
AIC_ss=AIC+(2*(numparam+2)*(numparam+3))/(numdata-numparam-3);#adjusted AIC

#For comparison between models
F=((SSR_ss-SSR_full)/constraints)/(SSR_full/(numdata-numparam-1))
Cp=SSR_ss/V_full-numdata+2*numparam

output.writelines('|3-piece interpolation+linear|\t'+str(m)+'+/-'+str(Sm)+'\t'+str(b)+'+/-'+str(Sb)+'\t'+ str(Rc_ss)+'\t'+str(AIC_ss)+ '\t'+str(F)+'\t'+str(Cp)+'\t'+str(breaks_full)+'\n')
#print('|Full model|',m,'+/-',Sm,'|',b,'+/-',Sb,'|', Rc_full, AIC_full, ' - ',Cp,breaks_full)

plt.plot(x,y,'o',x,yHat_full,'-')
plt.show();

"""
###2-piece models
"""
"""
###two-piece linear
"""
#Basic information
numseg=2
numparam=4
constraints=6-numparam

#Regression
p_ss, yHat_ss, breaks_ss=TwoPiecewiseLinearModel(x,y)

#Compute model parameters and uncertainties
m=[(p_ss[1]-p_ss[0])/(breaks_ss[1]-breaks_ss[0]),(p_ss[2]-p_ss[1])/(breaks_ss[2]-breaks_ss[1])]
b=[p_ss[0],p_ss[1]]

#Regression Analysis
#Residual computation
Sm=[0,0];
Sb=[0,0];
sepDataX, sepDataY=seperateData(x,y,breaks_ss)
sepDataX, sepHatY=seperateData(x,yHat_ss,breaks_ss)
for i in range(numseg):
    num=len(sepDataX[i])
    x_mean=sum(sepDataX[i])/len(sepDataX[i])
    Sy=sum((sepDataY[i]-sepHatY[i])**2)
    Sx=sum((sepDataX[i]-x_mean)**2)
    SS=sum(sepDataX[i]**2)
    Sm[i]=Sy/(Sx*(num-2))
    Sb[i]=Sy*SS/(Sx*num*(num-2))
SSR_ss=res_squared(y,yHat_ss)

#R squared
R=1-SSR_ss/sum((y-y_mean)**2)
Rc_ss=R-(numparam*(1-R)/(numdata-numparam-1)); #adjusted R-squared

#AIC
V_ss=SSR_ss/(numdata-numparam-1)
AIC=numdata*np.log(V_ss)+numdata*np.log(float(numdata-numparam-1)/numdata)+2*numdata+4;
AIC_ss=AIC+(2*(numparam+2)*(numparam+3))/(numdata-numparam-3);#adjusted AIC

#For comparison between models
F=((SSR_ss-SSR_full)/constraints)/(SSR_full/(numdata-numparam-1))
Cp=SSR_ss/V_full-numdata+2*numparam

output.writelines('|2-piecewise linear model|\t'+str(m)+'+/-'+str(Sm)+'\t'+str(b)+'+/-'+str(Sb)+'\t'+ str(Rc_ss)+'\t'+str(AIC_ss)+ '\t'+str(F)+'\t'+str(Cp)+'\t'+str(breaks_ss)+'\n')
#print('|2-piecewise linear model|',m,'+/-',Sm,'|',b,'+/-',Sb,'|', Rc_ss, AIC_ss, F , Cp,breaks_ss)

plt.plot(x,y,'o',x,yHat_ss,'-')
plt.show();
"""
###two-piece interpolation and linear
"""
#Basic information
numparam=3;
constraints=6-numparam

#Regression
slope,intercept, yHat_ss, breaks_ss=TwoPieceInterModel(x,y)

#Regression Analysis
#Residual computation
sepDataX, sepDataY=seperateData(x,y,breaks_ss)
sepDataX, sepHatY=seperateData(x,yHat_ss,breaks_ss)
for i in range(numseg-1):
    num=len(sepDataX[i+1])
    x_mean=sum(sepDataX[i+1])/len(sepDataX[i+1])
    Sy=sum((sepDataY[i+1]-sepHatY[i+1])**2)
    Sx=sum((sepDataX[i+1]-x_mean)**2)
    SS=sum(sepDataX[i+1]**2)
    Sm=Sy/(Sx*(num-2))
    Sb=Sy*SS/(Sx*num*(num-2))
SSR_ss=res_squared(y,yHat_ss)

#R-squared
R=1-SSR_ss/sum((y-y_mean)**2)
Rc_ss=R-(numparam*(1-R)/(numdata-numparam-1));#adjusted R-squared

#AIC
V_ss=SSR_ss/(numdata-numparam-1)
AIC=numdata*np.log(V_ss)+numdata*np.log(float(numdata-numparam-1)/numdata)+2*numdata+4;
AIC_ss=AIC+(2*(numparam+2)*(numparam+3))/(numdata-numparam-3); #adjusted AIC

#For comparison between models
F=((SSR_ss-SSR_full)/constraints)/(SSR_full/(numdata-numparam-1))
Cp=SSR_ss/V_full-numdata+2*numparam

output.writelines('|interpolation+linear model|\t'+str(slope)+'+/-'+str(Sm)+'\t'+str(intercept)+'+/-'+str(Sb)+'\t'+ str(Rc_ss)+'\t'+str(AIC_ss)+ '\t'+str(F)+'\t'+str(Cp)+'\t'+str(breaks_ss)+'\n')
#print('|2-piece interpolation+linear model|',slope,'+/-',Sm,'|',intercept,'+/-',Sb,'|', Rc_ss, AIC_ss, F , Cp,breaks_ss)

plt.plot(x,y,'o',x,yHat_ss,'-')
plt.show();
"""
##single linear fit
"""
#Basic information
numparam=2;
constraints=6-numparam;
#Regression
slope, intercept, r_value, p_value, std_err=stats.linregress(x,y);
yHat_ss=slope*x+intercept;

#Regression Analysis
#Residual computation
x_mean=sum(x)/len(x)
Sy=sum((y-yHat_ss)**2)
Sx=sum((x-x_mean)**2)
SS=sum(x**2)
Sm=Sy/(Sx*(num-2))
Sb=Sy*SS/(Sx*num*(num-2))
SSR_ss=res_squared(y,yHat_ss)

#R-squared
R=1-SSR_ss/sum((y-y_mean)**2)
Rc_ss=R-(numparam*(1-R)/(numdata-numparam-1));#adjusted R-squared

#AIC
V_ss=SSR_ss/(numdata-numparam-1)
AIC=numdata*np.log(V_ss)+numdata*np.log(float(numdata-numparam-1)/numdata)+2*numdata+4;
AIC_ss=AIC+(2*(numparam+2)*(numparam+3))/(numdata-numparam-3);#adjusted AIC

#For comparison between models
F=((SSR_ss-SSR_full)/constraints)/(SSR_full/(numdata-numparam-1))
Cp=SSR_ss/V_full-numdata+2*numparam

output.writelines('|linear model|\t'+str(slope)+'+/-'+str(Sm)+'\t'+str(intercept)+'+/-'+str(Sb)+'\t'+ str(Rc_ss)+'\t'+str(AIC_ss)+ '\t'+str(F)+'\t'+str(Cp)+'\t -'+'\n')
#print('|linear model|',slope,'+/-',Sm,'|',intercept,'+/-',Sb,'|', Rc_ss, AIC_ss, F , Cp,'-')

output.close()#close output file

plt.plot(x,y,'o',x,yHat_ss,'-')
plt.show();