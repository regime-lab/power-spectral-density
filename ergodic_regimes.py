import numpy as np
import matplotlib.pyplot as plt

def generate_time_series(length, autocorr_decays):
    """
    Generate a time series with long-range dependence.

    Parameters:
    - length: int, length of the time series
    - autocorr_decays: float array, decay factor controlling long-range dependence

    Returns:
    - time_series: numpy array, generated time series
    """
    time_series = np.zeros(length)
    for i in range(1, length):
        decay = autocorr_decays[i] if i < len(autocorr_decays) else 0.
        time_series[i] = decay * time_series[i-1] + np.random.normal(loc=0, scale=1)
    return time_series

def calculate_auto_correlation(time_series):
    """
    Calculate auto-correlation of a time series.

    Parameters:
    - time_series: numpy array, input time series

    Returns:
    - auto_corr: numpy array, auto-correlation values
    """
    length = len(time_series)
    auto_corr = np.correlate(time_series, time_series, mode='full')
    auto_corr = auto_corr[length-1:]
    auto_corr /= auto_corr[0]  # normalize by the first value
    return auto_corr

def plot_auto_correlation(auto_corr):
    """
    Plot the auto-correlation values.

    Parameters:
    - auto_corr: numpy array, auto-correlation values
    """
    plt.plot(auto_corr)
    plt.xlabel('Lag')
    plt.ylabel('Auto-correlation')
    plt.title('Auto-correlation of Time Series')
    plt.grid(True)
    plt.show()

LENGTH = 160

def compare_averages(time_series, window_size, autocorr_decays):
    """
    Compare ensemble average and time-domain average using windows of the data.

    Parameters:
    - time_series: numpy array, input time series
    - window_size: int, size of the window for averaging
    """
    length = len(time_series)
    num_windows = length // window_size
    ensemble_avg = np.zeros(num_windows)
    time_domain_avg = np.zeros(num_windows)
    local_time_series = generate_time_series(LENGTH, autocorr_decays)
    
    for i in range(num_windows):    
    
        ensemble_trials = 100
        for j in range(ensemble_trials): 
            ensemble_avg[i] += np.mean(generate_time_series(window_size, autocorr_decays))
        ensemble_avg[i] = ensemble_avg[i]/ensemble_trials    
        time_domain_avg[i] = np.mean(local_time_series[:i*window_size + window_size])
    
    return num_windows, ensemble_avg, time_domain_avg, local_time_series

# Generate a time series with long-range dependence after wait period 
rough_autocorr= [   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.6, 0.6, 0.6, 0.6,
                       0.5, 0.4, 0.7, 0.7, 0.6,
                       0.6, 0.6, 0.6, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 
                       0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 
                       0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
                       0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
                       0.3, 0.3, 0.3, 0.2, 0.1, 0.1 ]
                       
#rough_autocorr=[0]
print(len(rough_autocorr))
time_series = generate_time_series(LENGTH, rough_autocorr)

# Calculate auto-correlation
auto_corr = calculate_auto_correlation(time_series)

# Plot auto-correlation
plot_auto_correlation(auto_corr[1:])

# Compare ensemble average and time-domain average
window_size = 40
num_windows = None
ensemble_avg = None
time_domain_avg = None
local_time_series = None

fig,axs=plt.subplots(2)
for excursion_pattern in range(window_size):
    num_windows, ensemble_avg, time_domain_avg, local_time_series = compare_averages(time_series, window_size, rough_autocorr)
    axs[0].plot(range(num_windows), time_domain_avg, label='Time-Domain Average', color='darkorange', alpha=0.25)
    
axs[1].plot(range(num_windows), ensemble_avg, label='Ensemble Average', alpha=0.5)
plt.xlabel('Window Index')
plt.ylabel('Average')
axs[0].legend(['Excursion Patterns (LRD time-domain)'])
axs[1].legend()
plt.grid(True)
plt.show()

import gpytorch
import torch 
import seaborn as sns 
from sklearn.cluster import KMeans

# Evaluate kernel self similarity matrix (aka 'affinity matrix') 
kernel = gpytorch.kernels.RBFKernel(lengthscale=10)
C = (kernel(torch.tensor(local_time_series)).evaluate()).detach().numpy() 


# Define the number of random Fourier features
num_features = 3

# Generate random Fourier frequencies (random projection matrix)
from sklearn.random_projection import GaussianRandomProjection
random_projection = GaussianRandomProjection(n_components=num_features)
random_projection.fit(C)

# Apply the random projection to the affinity matrix
rff_approximation = random_projection.transform(C)
sns.lineplot(data=rff_approximation)
plt.show()
print(rff_approximation)


eigenvalues, eigenvectors = np.linalg.eig(C)
# Cluster eigenvalues (TODO Spectral clustering + Random Fourier Features + Wavelet connections)
#v0 = [float(x) for x in eigenvectors[:, 0]]
#v1 = [float(x) for x in eigenvectors[:, 1]]

#v2 = [float(x) for x in eigenvectors[:, 2]]
import pandas as pd 
#featuredf = pd.DataFrame()
#featuredf['x0']=v0
#featuredf['x1']=v1
#featuredf['x2']=v2
kmeans_n=2
kmeans_lbl = KMeans(n_clusters=kmeans_n).fit(rff_approximation).labels_
fig,ax=plt.subplots()
#sns.scatterplot(data=v0,s=3.5,ax=ax)
#sns.scatterplot(data=v1,s=3.5,ax=ax)
#sns.scatterplot(data=v2,s=3.5,ax=ax)

sns.lineplot(data=local_time_series, ax=ax)
state_counts = np.zeros(kmeans_n)
for M1 in kmeans_lbl:
  state_counts[M1] += 1 
  
for M2 in range(len(kmeans_lbl)): 
  if kmeans_lbl[M2] == np.argmin(state_counts):
    ax.axvline(M2, color='black', alpha=0.15)
plt.show()


# Plot the evaluation results
fig, ax = plt.subplots()
im = ax.imshow(C, cmap='viridis', origin='lower')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('measure', rotation=-90, va="bottom")

# Set labels
ax.set_xlabel('time')
ax.set_ylabel('time')
ax.set_title('Affinity Matrix')

# Show the plot
plt.grid(False)
plt.show()

