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

def compare_averages(time_series, window_size):
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
        window = time_series[i*window_size : (i+1)*window_size]
        ensemble_avg[i] = np.mean(window)
        time_domain_avg[i] = np.mean(time_series[:i*window_size + window_size])

    plt.plot(range(num_windows), ensemble_avg, label='Ensemble Average')
    plt.plot(range(num_windows), time_domain_avg, label='Time-Domain Average')
    plt.xlabel('Window Index')
    plt.ylabel('Average')
    plt.title('Comparison of Ensemble Average and Time-Domain Average')
    plt.legend()
    plt.grid(True)
    plt.show()


# Generate a time series with long-range dependence
length = 1000
alpha_decay =[ 0 ] #0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1 ] 
time_series = generate_time_series(length, alpha_decay)

# Calculate auto-correlation
auto_corr = calculate_auto_correlation(time_series)

# Plot auto-correlation
plot_auto_correlation(auto_corr)

# Compare ensemble average and time-domain average
window_size = 50
compare_averages(time_series, window_size)
