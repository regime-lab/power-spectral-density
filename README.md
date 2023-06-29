# power-spectral-density

The spectral density and ergodicity of a time series are related. We can demonstrate this by constructing a data generating process where a latent effect of auto-correlation diffuses away slowly over time, which is also known as long-range dependence (LRD). The frequency of the distinct variance levels in the data corresponds to how a large shock in the variance tends to be followed by a diffusing process of smaller and smaller shocks until the effect is gone or overcome by a new shock. 

The "Excursion Pattern (LRD)" shown below in the top panel is the time-domain representation of the diffusion process. Since eventually the auto-correlation decays away fully the time-domain representation settles on a value quite different than the ensemble average. This doesn't happen in the noise process (without LRD) because there is no initial decaying effect for the ensemble averaging step and initial phase of the time-domain to get 'stuck' in. This is like a temporary stable state that diffuses away and can only be observed if you let the data generating process run past the length of the LRD. 

RBF Kernels: 
https://regimelab.substack.com/p/kernels-and-attention-mechanism

Wavelets:
https://en.wikipedia.org/wiki/Wavelet

Ergodicity: 
https://regimelab.substack.com/p/ergodic-regimes

With LRD:
<img width="1068" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/0bcf8c16-d6b2-4de7-88a0-ace9a51d0560">

Without LRD: 
<img width="1009" alt="image" src="https://github.com/regime-lab/power-spectral-density/assets/114866071/cec9ce0a-2c58-42c7-a22d-cf7338df63ca">
