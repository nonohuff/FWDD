# FWDD
FWDD (Finite Width Dynamical Decoupling) implements Dynamical Decoupling considering the effects of FInite Pulse Widths. All times in the tutorials use microseconds $(\mu s)$. 

The code presented here is used in the paper [Quantum sensing with a spin ensemble in a two-dimensional material](http://arxiv.org/abs/2509.08984) to predict noise spectra.

# Coherence-Noise Relationship

A given dynamical decoupling pulse sequence of $N$ pulses (e.g. CPMG, XY8) yields a corresponding filter function, $F_N(\omega, t)$, which is related to the coherence decay as follows:

$$
\begin{align}
C_N(t) = e^{-\chi_N(t)}, \quad \chi(t) = (t/T_{2})^\beta, \\
\chi_N(t) = \frac{1}{\pi} \int _{0}^{\infty} d\omega S(\omega) \frac{F_N(\omega,t)}{\omega^2}
\end{align}
$$

Here, $C_N(t)$ is the coherence decay as a function of time characterized by two important parameters: $T_2$ - coherence time and $\beta$ - stretch factor. $F_N(\omega,t)$ is a filer function defined by the dynamical decoupling pulse sequence, and $S(\omega)$ is the power spectral density of the underlying noise (standard units $\frac{Hz^2}{Hz}$). 

## Delta Function Approximation

If $F_N(\omega,t)$ is approximated as a $\delta$ function peaked at $\omega_0 =\pi N/T$, where the total experiment time, $T=2N\tau+Nt_{\pi}$, $2\tau$ being the wait time between the pulses and $t_{\pi}$ is the pulse width. Then, the integral above is trival to invert and we have:

$$
\begin{equation}
    S(\omega) = -\pi \frac{ln(C_N(T))}{T}.
\end{equation}
$$

This approximation holds true when the $\pi$-pulses are themselves much shorter compared to the delay between them and as a consequence the coherence time, $T_2$.

## Finite-width Pulses

When the length of the $\pi$-pulse becomes a sizable fraction of the delay (or $T_2$), such a $\delta$ approximation cannot be made. In that case, the filter function needs to be modified to accommodate for the finite pulse duration and can be expressed as:

$$
F_{N}(\omega,T)=\left|1+(-1)^{N+1}e^{i\omega T} +2\sum_{k=1}^N(-1)^ke^{i\omega t_{k}}\cos\left(\frac{\omega t_{\pi}}{2}\right)\right|^{2}
$$

Where $t_k$ is the time corresponding to the center of the $k^{th}$ pulse, and $t_{\pi}$ is the pulse width.

# Tutorial Notebooks

We have included several tutorial notebooks to make adapting this code to your purposes easier. They cover how we implement noise spectral densities, $S(\omega)$, the finite-width filter function, coherence profile, $C_N(t)$, and finally, how we go about fitting noise profiles to observed $C_N(t)$ data (and how to do it for you own data/noise models!). The reccomended viewwing of these notebooks are. 

`noise_tutorial.ipynb` -> `filter_function_tutorial.ipynb` -> `coherence_profile_tutorial.ipynb` -> `noise_learning_fitting.ipynb`

# Conventions
* $\omega = 2 \pi f$
* $t = \frac{1}{f}$
* Time varibles assume microseconds, so `tau_p = 0.024` means $0.024 \mu s$ or $24 ns$ 