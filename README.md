# Ergodic versus non-Ergodic Latent States 

The spectral density and ergodicity of a time series are related. We can demonstrate this by constructing a data generating process where a latent effect of auto-correlation diffuses away slowly over time, which is also known as long-range dependence (LRD). The frequency of the distinct variance levels in the data corresponds to how a large shock in the variance tends to be followed by a diffusing process of smaller and smaller shocks until the effect is gone or overcome by a new shock. 

The "Excursion Pattern (LRD)" shown below in the top panel is the time-domain representation of the diffusion process. Since eventually the auto-correlation decays away fully the time-domain representation settles on a value quite different than the ensemble average which repeats many trials of finite length and they always start from initial conditions. This doesn't happen in the noise process, without the LRD effect, because there is no initial decaying effect for the ensemble averaging step and initial phase of the time-domain to get 'stuck' in. This is like a temporary stable state that diffuses away and can only be observed if you let the data generating process run past the length of the LRD. 

Substack: https://regimelab.substack.com/p/ergodic-regimes

See <b>ergodic_regimes.py</b>

## Stable states at different areas of DGP in time domain

<img width="500" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/a4e4912f-35c3-40b7-af63-75127d6934b7">

## Auto-correlation decay rate examples (wave-like)

<img width="400" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/eb587428-b045-4207-a60b-e92857a10a1f"><br/>

<img width="400" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/2fd1fa9e-7476-4d88-bd36-240f3be0a8f3">

Squared 1st-order Diffs: 

<img width="400" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/50e8e43d-28ab-43f3-83ea-7711fe518434">

## RBF Kernels

RBF Kernels can be used to learn the LRD and rate of auto-correlation decay at different offsets and lag time horizons / long and short-term memory effects. We can cluster in the high-dimensional space generated by the RBF Kernel using the eigenvectors or Random Fourier Features (TBD). 

https://regimelab.substack.com/p/kernels-and-attention-mechanism

https://portfoliooptimizer.io/blog/correlation-matrices-denoising-results-from-random-matrix-theory/

The shaded latent states below are labeled using KMeans which clusters in the lower dimensional space of eigenvectors of the RBF Kernel matrix ie. affinity matrix.

<img width="500" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/04940d03-012d-49ad-81a9-1f9851b75795">

<img width="500" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/ee2c3e71-b4c9-4aa3-a8bf-243e492b76b7"><br/>

<img width="500" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/7f233e28-feef-4786-9162-e09110a6f59a">

## RBF Kernel w/ Squared Diffs (Rolling Volatility) 

https://stats.stackexchange.com/questions/386813/use-the-rbf-kernel-to-construct-a-positive-definite-covariance-matrix

Applying the kernel to the squared return difference (1st-order differencing) is one way to model volatility. 

<img width="500" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/097dadac-ca5d-43a3-b194-ba4e84fd222e">

<img width="500" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/97b519a9-3db7-43da-98ec-ce8a81f51e9d">

## Why This is Useful 

Wavelets: 

The goal of this mini-project will be to see if wavelets/or at least kernels can be used to describe the occurrence of LRD latent states in the data that slowly diffuse away and then may repeat again, periodically. This can be used to design features in many domains such as neuroscience, economics, climate modeling where there is a multifractal scaling aspect to the data and the frequencies of the power spectrum are time-varying: non-stationary, non-ergodic time series. 

What we are finding is periods of time/phases over windows in the time-varying domain where the time series is self-similar versus periods where it is less self-similar or not self-similar, it could be anti-persistent and mean reverting. This is useful because it tells us something unique that instantaneous statistical moments, such as mean or variance in a hidden Markov model cannot tell us. By preserving a notion of order and self-similarity this can also capture path dependence. 

Links:

https://en.wikipedia.org/wiki/Wavelet

https://archive.physionet.org/tutorials/multifractal/wavelet.htm
