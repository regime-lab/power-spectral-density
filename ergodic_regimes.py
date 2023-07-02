import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import gpytorch
import torch 
import seaborn as sns 

from sklearn.cluster import KMeans

    
def generate_time_series(length, autocorr_decays):
    """
    Generate a time series with LRD using autocorr_decays array. 
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
    auto_corr /= auto_corr[0]  
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

LENGTH = 100

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
rough_autocorr= [      0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                       0.6, 0.6, 0.6, 0.6, 0.5, 0.4, 
                       0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 
                       0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 
                       0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 
                       0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
                       0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 
                       0.1, 0.1 ]
                       
#rough_autocorr=[0]
print(len(rough_autocorr))
time_series = generate_time_series(LENGTH, rough_autocorr)

# Calculate auto-correlation
auto_corr = calculate_auto_correlation(time_series)

# Plot auto-correlation
plot_auto_correlation(auto_corr[1:])

# Compare ensemble average and time-domain average
window_size = 10
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

for step in range(1, int(len(local_time_series) / 10)): 
        
    # Evaluate kernel self similarity matrix (aka 'affinity matrix') as it grows with data 
    kernel = gpytorch.kernels.RBFKernel(lengthscale=10)
    C = (kernel(torch.tensor(local_time_series[:step*10])).evaluate()).detach().numpy() 

    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # Cluster eigenvalues (TODO Spectral clustering + Random Fourier Features + Wavelet connections)
    v0 = [float(x) for x in eigenvectors[:, 0]]
    v1 = [float(x) for x in eigenvectors[:, 1]]

    featuredf = pd.DataFrame()
    featuredf['x0']=v0
    featuredf['x1']=v1
    kmeans_n=2
    kmeans_lbl = KMeans(n_clusters=kmeans_n).fit(featuredf).labels_
    fig,ax=plt.subplots()

    sns.lineplot(data=np.cumsum(local_time_series[:step*10]), ax=ax)
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
