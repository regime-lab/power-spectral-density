# Ergodic versus non-Ergodic Latent States 

The spectral density and ergodicity of a time series are related. We can demonstrate this by constructing a data generating process where a latent effect of auto-correlation diffuses away slowly over time, which is also known as long-range dependence (LRD). The frequency of the distinct variance levels in the data corresponds to how a large shock in the variance tends to be followed by a diffusing process of smaller and smaller shocks until the effect is gone or overcome by a new shock. 

The "Excursion Pattern (LRD)" shown below in the top panel is the time-domain representation of the diffusion process. Since eventually the auto-correlation decays away fully the time-domain representation settles on a value quite different than the ensemble average which repeats many trials of limited length so they are always in the LRD (non-ergodic) regime, as they start from initial conditions. This doesn't happen in the noise process (without LRD) because there is no initial decaying effect for the ensemble averaging step and initial phase of the time-domain to get 'stuck' in. This is like a temporary stable state that diffuses away and can only be observed if you let the data generating process run past the length of the LRD. 

Substack: https://regimelab.substack.com/p/ergodic-regimes

See <b>ergodic_regimes.py</b>

## Stable state at beginning of DGP
<img width="1440" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/a205685b-b426-434d-9b38-8d99d77cb3be">

## Stable state in middle of DGP (wave-like): 
<img width="1440" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/0e5bc19c-cb58-4c55-94cd-deecc95a3ac0">

## Wave-like Shock: 
<img width="1440" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/a7be0e5f-5e7c-4008-8649-08b576610eb8">

## RBF Kernels

RBF Kernels can be used to learn the LRD and rate of auto-correlation decay at different offsets and lag time horizons / long and short-term memory effects. 

https://regimelab.substack.com/p/kernels-and-attention-mechanism

<img width="690" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/d9769cb8-eed4-4ff9-b239-15210831f663">


<img width="292" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/b6144bdc-9631-40a6-837b-f73f76a114d6">

## Wavelets

The goal of this mini-project will be to see if wavelets/kernels can be used to describe the occurrence of LRD latent states in the data that slowly diffuse away and then may repeat again, periodically. This can be used to design features in many domains e.g. neuroscience, financial markets, climate modeling where there is a multifractal scaling aspect to the data and the frequencies of the power spectrum are time-varying: non-stationary, non-ergodic time series. 

Links:

https://en.wikipedia.org/wiki/Wavelet

https://archive.physionet.org/tutorials/multifractal/wavelet.htm
