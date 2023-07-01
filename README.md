# Ergodic versus non-Ergodic Latent States 

The spectral density and ergodicity of a time series are related. We can demonstrate this by constructing a data generating process where a latent effect of auto-correlation diffuses away slowly over time, which is also known as long-range dependence (LRD). The frequency of the distinct variance levels in the data corresponds to how a large shock in the variance tends to be followed by a diffusing process of smaller and smaller shocks until the effect is gone or overcome by a new shock. 

The "Excursion Pattern (LRD)" shown below in the top panel is the time-domain representation of the diffusion process. Since eventually the auto-correlation decays away fully the time-domain representation settles on a value quite different than the ensemble average which repeats many trials of finite length and they always start from initial conditions. This doesn't happen in the noise process, without the LRD effect, because there is no initial decaying effect for the ensemble averaging step and initial phase of the time-domain to get 'stuck' in. This is like a temporary stable state that diffuses away and can only be observed if you let the data generating process run past the length of the LRD. 

Substack: https://regimelab.substack.com/p/ergodic-regimes

See <b>ergodic_regimes.py</b>

## Stable state in middle of DGP (wave-like): 
<img width="625" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/f34667fd-a6c4-416f-b6f4-e034e8ca28f6">

## Wave-like oscillation: 
<img width="625" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/7e238277-8555-4df2-bfdf-dd61a733fa63">

## Slow LRD Decay
<img width="625" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/04d981bf-3b2d-4e81-acab-65622d0a0a62">

## RBF Kernels

RBF Kernels can be used to learn the LRD and rate of auto-correlation decay at different offsets and lag time horizons / long and short-term memory effects. 

https://regimelab.substack.com/p/kernels-and-attention-mechanism

<img width="591" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/edc3a330-1257-4ddf-b209-197ebca36949">

## Wavelets (TBD)

The goal of this mini-project will be to see if wavelets/or at least kernels can be used to describe the occurrence of LRD latent states in the data that slowly diffuse away and then may repeat again, periodically. This can be used to design features in many domains e.g. neuroscience, financial markets, climate modeling where there is a multifractal scaling aspect to the data and the frequencies of the power spectrum are time-varying: non-stationary, non-ergodic time series. 

Links:

https://en.wikipedia.org/wiki/Wavelet

https://archive.physionet.org/tutorials/multifractal/wavelet.htm
