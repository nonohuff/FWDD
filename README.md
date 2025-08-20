# FWDD
FWDD (Finite Width Dynamical Decoupling) implements Dynamical Decoupling considering the effects of FInite Pulse Widths. 

# Coherence-Noise Relationship

A given dynamical decoupling pulse sequence of $N$ pulses (e.g. CPMG, XY8) yields a corresponding filter function, $F_N(\omega, t)$, which is related to the coherence decay as follows:

$$
\begin{align}
C(t) = e^{-\chi(t)}, \quad \chi(t) = (t/T_{2})^\beta, \\
\chi(t) = \frac{1}{\pi} \int _{0}^{\infty} d\omega S(\omega) \frac{F_N(\omega,t)}{\omega^2}
\end{align}
$$

Here, $C(t)$ is the coherence decay as a function of time characterized by two important parameters: $T_2$ - coherence time and $\beta$ - stretch factor. $F_N(\omega,t)$ is a filer function defined by the dynamical decoupling pulse sequence, and $S(\omega)$ is the spectral density of the underlying noise. 

## Delta Function Approximation

If $F_N(\omega,t)$ is approximated as a $\delta$ function peaked at $\omega_0 =\pi N/T$, where the total experiment time, $T=2N\tau+Nt_{\pi}$, $2\tau$ being the wait time between the pulses and $t_{\pi}$ is the pulse width. Then, the integral above is trival to invert and we have:

$$
\begin{equation}
    S(\omega) = -\pi \frac{ln(C(T))}{T}.
\end{equation}
$$

This approximation holds true when the $\pi$-pulses are themselves much shorter compared to the delay between them and as a consequence the coherence time, $T_2$.

## Finite-width Pulses

When the length of the $\pi$-pulse becomes a sizable fraction of the delay (or $T_2$), such a $\delta$ approximation cannot be made. In that case, the filter function needs to be modified to accommodate for the finite pulse duration and can be expressed as:

$$
F_{N}(\omega,T)=\left|1+(-1)^{n+1}e^{i\omega T} +2\sum_{k=1}^N(-1)^ke^{i\omega t_{k}}\cos\left(\frac{\omega t_{\pi}}{2}\right)\right|^{2}
$$

Where $t_k$ is the time corresponding to the center of the $k^{th}$ pulse, and $t_{\pi}$ is the pulse width.
