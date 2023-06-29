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

    for i in range(num_windows):
        ensemble_trials = 100
        for j in range(ensemble_trials): 
            ensemble_avg[i] += np.mean(generate_time_series(window_size, alpha_decays))
        ensemble_avg[i] = ensemble_avg[i]/ensemble_trials
        
        time_domain_avg[i] = np.mean(time_series[:i*window_size + window_size])

    plt.plot(range(num_windows), ensemble_avg, label='Ensemble Average', alpha=0.5)
    plt.plot(range(num_windows), time_domain_avg, label='Time-Domain Average')
    plt.xlabel('Window Index')
    plt.ylabel('Average')
    plt.title('Comparison of Ensemble Average and Time-Domain Average')
    plt.legend()
    plt.grid(True)
    plt.show()


# Generate a time series with long-range dependence
length = 10000
alpha_decays = [0.89,0.87,0.85,0.83,0.81,0.79,0.7,0.6,0.4,0.33,0.32,0.31] 
time_series = generate_time_series(length, alpha_decays)

# Calculate auto-correlation
auto_corr = calculate_auto_correlation(time_series)

# Plot auto-correlation
plot_auto_correlation(auto_corr[1:])

# Compare ensemble average and time-domain average
window_size = 100
compare_averages(time_series, window_size, alpha_decays)
