import numpy as np
import matplotlib.pyplot as plt

def generate_time_series(length, alpha_decays):
    """
    Generate a time series with long-range dependence.

    Parameters:
    - length: int, length of the time series
    - alpha_decays: float array, decay factor controlling long-range dependence

    Returns:
    - time_series: numpy array, generated time series
    """
    time_series = np.zeros(length)
    for i in range(1, length):
        decay = alpha_decays[i] if i < len(alpha_decays) else 0.
        time_series[i] = decay * time_series[i-1] + np.random.normal(loc=1, scale=2)
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

LENGTH = 3000

def compare_averages(time_series, window_size, alpha_decays):
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
    local_time_series = generate_time_series(LENGTH, alpha_decays)
    
    for i in range(num_windows):    
    
        ensemble_trials = 10
        for j in range(ensemble_trials): 
            ensemble_avg[i] += np.mean(generate_time_series(window_size, alpha_decays))
        ensemble_avg[i] = ensemble_avg[i]/ensemble_trials    
        time_domain_avg[i] = np.mean(local_time_series[:i*window_size + window_size])
    
    return num_windows, ensemble_avg, time_domain_avg

# Generate a time series with long-range dependence
alpha_decays = [    0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
                0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9,
                0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9,
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9
                ,0.9, 0.9, 0.9, 0.9, 0.9, 0.9
                ,0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1, 0.5, 0.6, 0.7,
                0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
#alpha_decays=[0]
print(len(alpha_decays))
time_series = generate_time_series(LENGTH, alpha_decays)

# Calculate auto-correlation
auto_corr = calculate_auto_correlation(time_series)

# Plot auto-correlation
plot_auto_correlation(auto_corr[1:])

# Compare ensemble average and time-domain average
window_size = 150
num_windows = None
ensemble_avg = None
time_domain_avg = None
 
fig,axs=plt.subplots(2)
for excursion_pattern in range(window_size):
    num_windows, ensemble_avg, time_domain_avg = compare_averages(time_series, window_size, alpha_decays)
    axs[0].plot(range(num_windows), time_domain_avg, label='Time-Domain Average', color='darkred', alpha=0.1)
    
axs[1].plot(range(num_windows), ensemble_avg, label='Ensemble Average', alpha=0.5)
plt.xlabel('Window Index')
plt.ylabel('Average')
axs[0].legend(['Excursion Patterns (LRD)'])
axs[1].legend()
plt.grid(True)
plt.show()
