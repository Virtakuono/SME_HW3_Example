#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.special as spsp
import numpy.linalg
import scipy.stats
import random

phi = lambda x: 0.5*(1+spsp.erf(x/np.sqrt(2)))
heaviside = lambda x: 0.5 * (np.sign(x) + 1)
stdDev = lambda x: np.sqrt(np.var(x))
# the two pi constant is insignificant and omitted
gaussDensity = lambda x,mu,sigma: np.exp((x-mu)**2/(2*sigma**2))/sigma
exponentialDensity = lambda x,lam: np.exp(-1*x/lam)/lam

### Problem 1

print('--Problem 1--')

# Naive MC estimation
for M in (10000,100000):
	randomSample = 2+sp.randn(M)
	g = heaviside(randomSample-5.75)
	estimator = np.mean(g)
	standardDeviation = np.sqrt(np.var(g)/M)
	d = standardDeviation*scipy.stats.norm.ppf(0.95)
	print('Sample size %d, 90 %% confidence interval for the MC estimator is [%f,%f] (%f %% relative error)'\
	%(M,estimator+d,estimator+d,d/estimator*100))

# More refined MC estimation
for M in (10000,100000):
	randomSample = 5.75+sp.randn(M)
	g = heaviside(randomSample-5.75)
	dd1 = 1.0*g
	g *= np.exp((randomSample-5.75)**2/2-(randomSample-2)**2/2)
	dd2 = 1.0*g
	estimator = np.mean(g)
	standardDeviation = np.sqrt(np.var(g)/M)
	d = standardDeviation*scipy.stats.norm.ppf(0.95)
	print('Sample size %d, 90 %% confidence interval for the MC estimator is [%f,%f] (%f %% relative error)'\
	%(M,estimator+d,estimator+d,d/estimator*100))

print('-----')

### Problem 3

print('--Problem 3--')

for M in (50,500,5000):
	randomSample = sp.randn(M)
	h = 20/np.sqrt(M)
	K = lambda x: 0.5*(1-heaviside(abs(x)-1))
	plot_N = 400
	approx_x = np.linspace(-2.5,2.5,plot_N)
	approx_y = 0*approx_x
	for m in range(plot_N):
		for m2 in range(M):
			approx_y[m] += K((approx_x[m]-randomSample[m2])/h)
	approx_y *= h
	approx_y /= M
	approx_y /= 400
	plt.figure()
	plt.hist(randomSample,normed=True)
	plt.plot(approx_x,approx_y*M,'k-',linewidth=2)
	plt.xlabel('$x$')
	plt.ylabel('$C \phi (x)$')
	plt.title('KDE and histogram with $M=%d$'%(M,))
	plt.grid(1)
	plt.savefig('./kde_histogram_M_%d.pdf'%(M,))

M =100

h = 0.5

# define whatever matrix to induce a bit of correlation	
coefficientMatrix = np.eye(2)
coefficientMatrix[0,1] += 0.3

randomSample = sp.randn(2,M)
randomSample = np.dot(coefficientMatrix,randomSample)

plot_N = 40
plot_x = np.linspace(-5,5,plot_N)
X,Y = np.meshgrid(plot_x,plot_x)
Z = np.zeros((plot_N,plot_N))
Z_ref = 1.0*Z

invCoefficient = numpy.linalg.inv(coefficientMatrix)

for row in range(plot_N):
	for col in range(plot_N):
		v = np.array([X[row,col],Y[row,col]])
		v = np.dot(invCoefficient,v)
		Z_ref[row,col] = np.exp(-np.dot(v,v)/2)
		norm = lambda v: np.dot(v,v)
		for m in range(M):
			Z[row,col] +=  K(norm([X[row,col]-randomSample[0,m], Y[row,col]-randomSample[1,m] ])/2)

Z[row,col] /= M
Z[row,col] *= h**2

plt.figure()
plt.contour(X,Y,Z)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('KDE Approximation of density, $M=%d$'%(M,))
plt.savefig('approximation.pdf')

plt.figure()
plt.contour(X,Y,Z_ref)
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Reference density, $M=%d$'%(M,))
plt.savefig('reference.pdf')

# b-part

print('-----')

### Problem 5

print('--Problem 5--')

# set parameters

sigma = 0.1
r = 0.05
V_0 = 100
N_t = 51
M = 10000
K = 115

pos = lambda x: np.fmax(x,np.zeros(len(x)))

gs = np.zeros(M)
gs_a = np.zeros(M)
cv = np.zeros(M)
cv_a = np.zeros(M)

for m in range(M):
	dWs = sp.randn(N_t-1)
	Rs = np.sqrt(sigma/50)*dWs
	Rs_a = -1*Rs
	cv[m] = np.dot(range(50,0,-1),dWs)
	cv_a[m] = np.dot(range(50,0,-1),-1*dWs)
	Rs += r/50
	Rs_a += r/50
	gs[m] = max(np.mean(V_0*np.exp(np.cumsum(Rs)))-K,0)
	gs_a[m] = max(np.mean(V_0*np.exp(np.cumsum(Rs_a)))-K,0)

cv_sig = np.sqrt(np.dot(range(50,0,-1),range(50,0,-1)))
P_otm = len(mlab.find(gs==0.0))/float(M)
K_cv = scipy.stats.norm.ppf(P_otm)
K_cv *= cv_sig
cv = pos(cv-K_cv)
cv_a = pos(cv_a-K_cv)
cvmean = cv_sig/np.sqrt(2*np.pi)*np.exp(-K_cv**2/2/cv_sig/cv_sig)
cvmean += K_cv*(phi(K_cv/cv_sig)-1)
cv -= cvmean
cv_a -= cvmean

rho = np.corrcoef(gs,cv)[0,1]

plt.figure()
plt.plot(gs,cv+cvmean,'kx')
plt.xlabel('MC Realisations')
plt.ylabel('Control variate realisations')
plt.title('Control variates vs MC realisations $\\rho=%.3f$'%(rho,))
plt.grid(1)
plt.savefig('cvplot.pdf')

beta = - rho*np.sqrt(np.var(gs)/np.var(cv))

estimator = np.mean(gs)
d = scipy.stats.norm.ppf(0.975)*stdDev(gs)/np.sqrt(M)
print('Plain vanilla MC - 95 %% confidence interval [%f,%f] - %f %% relative error - %f'%(estimator-d,estimator+d,d/estimator*100,d/estimator/2))

estimator = np.mean(np.concatenate((gs,gs_a)))
d = scipy.stats.norm.ppf(0.975)*stdDev(np.concatenate((gs,gs_a)))/np.sqrt(2*M)
print('Antithetic MC - 95 %% confidence interval [%f,%f] - %f %% relative error - %f'%(estimator-d,estimator+d,d/estimator*100,d/estimator/np.sqrt(2)))

estimator = np.mean(gs+beta*cv)
d = scipy.stats.norm.ppf(0.975)*stdDev((gs+beta*cv))/np.sqrt(M)
print('Control Variate MC - 95 %% confidence interval [%f,%f] - %f %% relative error - %f'%(estimator-d,estimator+d,d/estimator*100,d/estimator/np.sqrt(2)))

estimator = np.mean(np.concatenate((gs+beta*cv,gs_a+beta*cv_a)))
d = scipy.stats.norm.ppf(0.975)*stdDev(np.concatenate((gs+beta*cv,gs_a+beta*cv_a)))/np.sqrt(2*M)
print('Hybrid MC - 95 %% confidence interval [%f,%f] - %f %% relative error - %f'%(estimator-d,estimator+d,d/estimator*100,d/estimator*2))

print('-----')

### Problem 6

print('--Problem 6--')

# In this exercise, we use reuse the sample from before

N_resample = 1000
quantiles = (0.9,0.95,0.99)
realisations = []
q = 0.95

for n in range(N_resample):
	resample_indices = [random.randint(0,M-1) for foo in range(M)]
	new_sample = np.copy(gs[resample_indices])
	new_sample.sort()
	realisations.append([np.mean(new_sample[p*M:]) for p in quantiles])

realisations = np.array(realisations)
for p_ind in range(3):
	temporaryVector = realisations[:,p_ind]
	temporaryVector.sort()
	i_low = temporaryVector[int(((1-q)/2)*N_resample)]
	i_high = temporaryVector[int((q-((1-q)/2))*N_resample+1)]
	print('For p = %.3f %%, 95 %% confidence interval for expected shortfall: [%.4f, %.4f]'%(quantiles[p_ind]*100,i_low,i_high))
	
print('-----')	
	
