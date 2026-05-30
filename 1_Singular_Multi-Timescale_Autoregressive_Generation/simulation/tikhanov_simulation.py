"""
Non-Linear Singular Perturbation Simulation Core (Singular-SSM)

Implements the coupled Multi-Timescale Selective Resonance State-Space Model dynamics.
Defines SingularPerturbationSystem, TokenStream, and trackers to support full ISS
stability verification and figure generation.
"""

import numpy as np


class TokenStream:
    def __init__(self, d_hidden=128, d_semantic=64, seed=0):
        self.d_h = d_hidden
        rng = np.random.RandomState(seed)
        U, _, Vt = np.linalg.svd(rng.randn(d_hidden, d_semantic), full_matrices=False)
        self.W_out = U @ Vt
        self.A_sem = rng.randn(d_semantic, d_semantic) * 0.2 / np.sqrt(d_semantic)

    def generate(self, n_tokens, perturb_step=None, perturb_mag=5.0,
                 drift_std=0.03, noise_std=0.15, seed=0):
        dt = 1.0 / 50.0
        rng = np.random.RandomState(seed)
        d_s = self.A_sem.shape[0]
        s = rng.randn(d_s) * 0.1
        stream = np.zeros((n_tokens, self.d_h))
        times = np.zeros(n_tokens)

        for k in range(n_tokens):
            if perturb_step is not None and abs(k - perturb_step) < 3:
                s += perturb_mag * rng.randn(d_s)
            s = s + self.A_sem @ np.tanh(s) * dt + drift_std * rng.randn(d_s) * np.sqrt(dt)
            stream[k] = self.W_out @ s + noise_std * rng.randn(self.d_h)
            times[k] = k * dt
        return times, stream


class MonolithicTracker:
    def __init__(self, d_hidden=128, seed=0):
        self.d_h = d_hidden
        self.dt = 1.0 / 50.0
        rng = np.random.RandomState(seed)
        self.A = rng.randn(d_hidden, d_hidden) * 0.4 / np.sqrt(d_hidden)
        self.K_obs = 1.5
        self.noise_proc = 0.01

    def simulate(self, n_tokens, token_stream, seed=0):
        rng = np.random.RandomState(seed)
        h = rng.randn(self.d_h) * 0.1
        h_traj = np.zeros((n_tokens, self.d_h))
        lp_traj = np.zeros(n_tokens)

        for k in range(n_tokens):
            obs = token_stream[k]
            dh = self.A @ np.tanh(h) * self.dt
            dh += self.K_obs * (obs - h) * self.dt
            dh += self.noise_proc * rng.randn(self.d_h) * np.sqrt(self.dt)
            h = h + dh
            h_traj[k] = h.copy()
            lp_traj[k] = -np.mean((obs - h)**2)
        return h_traj, lp_traj


class SingularSSMTracker:
    def __init__(self, d_hidden=128, d_semantic=64, epsilon=0.04, seed=0):
        self.d_h = d_hidden
        self.d_s = d_semantic
        self.epsilon = epsilon
        self.dt = 1.0 / 50.0
        self.dt_cog = 0.5
        rng = np.random.RandomState(seed)
        self.K_surf = 1.0
        self.W_cog = rng.randn(d_semantic, d_hidden) * 0.2 / np.sqrt(d_hidden)
        U, _, Vt = np.linalg.svd(rng.randn(d_hidden, d_semantic), full_matrices=False)
        self.W_film = U @ Vt
        self.gamma_decay = 2.0
        self.K_I = 0.3
        self.K_contract = 0.15
        self.noise_proc = 0.01

    def simulate(self, n_tokens, token_stream, seed=0):
        rng = np.random.RandomState(seed)
        x = rng.randn(self.d_s) * 0.1
        y = rng.randn(self.d_h) * 0.1
        omega_integral = np.zeros(self.d_s)
        y_traj = np.zeros((n_tokens, self.d_h))
        lp_traj = np.zeros(n_tokens)
        conditioning_target = np.tanh(x)

        for k in range(n_tokens):
            # Dynamic Target Anchors Progression
            # Prevents resetting by scaling anchors based on active semantic window
            if k % int(self.dt_cog / self.dt) == 0:
                surface_info = np.tanh(y)
                dx = self.dt_cog * 0.05 * (self.W_cog @ surface_info)
                x = x + dx + 0.01 * rng.randn(self.d_s)
                conditioning_target = np.tanh(x)
                omega_integral = np.zeros(self.d_s)

            steps_since_cog = k % int(self.dt_cog / self.dt)
            t_elapsed = steps_since_cog * self.dt
            attenuation = np.exp(-self.gamma_decay * t_elapsed)
            Phi = conditioning_target * attenuation + self.K_I * omega_integral

            gamma = self.W_film @ Phi
            scale = 1.0 + np.tanh(gamma)
            y_mod = scale * y

            obs = token_stream[k]
            dy = (1.0 / self.epsilon) * (
                -self.K_contract * (y - y_mod)
                + self.K_surf * (obs - y)
            ) * self.dt
            dy += self.noise_proc * rng.randn(self.d_h) * np.sqrt(self.dt)
            y = y + dy

            omega_s = np.tanh(y[:self.d_s])
            omega_integral += omega_s * self.dt

            y_traj[k] = y_mod.copy()
            lp_traj[k] = -np.mean((obs - y_mod)**2)

        return y_traj, lp_traj


class ZOHTracker:
    def __init__(self, d_hidden=128, d_semantic=64, epsilon=0.04, seed=0, variant="rigid"):
        self.d_h = d_hidden
        self.d_s = d_semantic
        self.epsilon = epsilon
        self.dt = 1.0 / 50.0
        self.dt_cog = 0.5
        self.variant = variant
        rng = np.random.RandomState(seed)
        self.W_cog = rng.randn(d_semantic, d_hidden) * 0.2 / np.sqrt(d_hidden)
        U, _, Vt = np.linalg.svd(rng.randn(d_hidden, d_semantic), full_matrices=False)
        self.W_proj = U @ Vt
        self.K_surf = 1.0 if variant == "adaptive" else 0.0
        self.K_contract = 0.15
        self.noise_proc = 0.01

    def simulate(self, n_tokens, token_stream, seed=0):
        rng = np.random.RandomState(seed)
        x = rng.randn(self.d_s) * 0.1
        y = rng.randn(self.d_h) * 0.1
        y_traj = np.zeros((n_tokens, self.d_h))
        lp_traj = np.zeros(n_tokens)
        zoh_target = self.W_proj @ np.tanh(x)

        for k in range(n_tokens):
            if k % int(self.dt_cog / self.dt) == 0:
                surface_info = np.tanh(y)
                dx = self.dt_cog * 0.05 * (self.W_cog @ surface_info)
                x = x + dx + 0.01 * rng.randn(self.d_s)
                zoh_target = self.W_proj @ np.tanh(x)

            obs = token_stream[k]
            dy = (1.0 / self.epsilon) * (
                -self.K_contract * (y - zoh_target)
                + self.K_surf * (obs - y)
            ) * self.dt
            dy += self.noise_proc * rng.randn(self.d_h) * np.sqrt(self.dt)
            y = y + dy

            y_traj[k] = y.copy()
            lp_traj[k] = -np.mean((obs - y)**2)

        return y_traj, lp_traj


class SingularPerturbationSystem:
    """Unified system class representing the fast-slow State-Space dynamics."""
    def __init__(self, d_c=64, d_m=128, epsilon=0.04, dt_cognitive=0.5, f_token=50.0):
        self.d_c = d_c
        self.d_m = d_m
        self.epsilon = epsilon
        self.dt_cog = dt_cognitive
        self.dt = 1.0 / f_token
        self.alpha_K = 0.02
        self.alpha_k = 0.02
        self.L_target = 1.0

        rng = np.random.RandomState(42)
        # Initialize projections for the unified parameterization
        self.W_D = rng.randn(d_m, d_c) * 0.1 / np.sqrt(d_c)
        self.b_D = rng.randn(d_m) * 0.1
        self.W_S = rng.randn(d_m, d_m, d_c) * 0.05 / np.sqrt(d_c)
        self.b_S = rng.randn(d_m, d_m) * 0.05
        
        self.B_0 = rng.randn(d_m, d_m) * 0.1 / np.sqrt(d_m)
        self.W_B = rng.randn(d_m, d_c) * 0.1 / np.sqrt(d_c)
        
        # Apply spectral norm clipping to W_B to satisfy Lipschitz limit L_target
        U_b, S_b, Vt_b = np.linalg.svd(self.W_B, full_matrices=False)
        S_b = np.clip(S_b, 0.0, self.L_target)
        self.W_B = U_b @ np.diag(S_b) @ Vt_b

        self.W_c = rng.randn(d_c, d_m) * 0.2 / np.sqrt(d_m)
        U, S, Vt = np.linalg.svd(self.W_c, full_matrices=False)
        S = np.clip(S, 0.0, self.L_target)
        self.W_c = U @ np.diag(S) @ Vt

        self.K_surf = 1.0
        self.K_contract = 0.15
        self.gamma_decay = 2.0
        self.K_I = 0.3
        self.noise_proc = 0.01

    def compute_A_fast(self, Phi):
        # R_D diagonal projection
        R_D = self.W_D @ Phi[:self.d_c] + self.b_D
        # D_ii = alpha_K + softplus(R_D_i)
        # softplus(x) = ln(1 + e^x)
        D_diag = self.alpha_K + np.log(1.0 + np.exp(np.clip(R_D, -20, 20)))
        D = np.diag(D_diag)
        
        # R_S skew-symmetric projection
        R_S = np.tensordot(self.W_S, Phi[:self.d_c], axes=(2, 0)) + self.b_S
        S = 0.5 * (R_S - R_S.T)
        
        # A_fast = -D + S
        return -D + S

    def compute_B_fast(self, Phi):
        # B_fast = B_0 + diag(W_B @ Phi)
        diag_W_B = self.W_B @ Phi[:self.d_c]
        return self.B_0 + np.diag(diag_W_B)

    def fast_dynamics(self, y, t, x_fixed, Phi):
        A_fast = self.compute_A_fast(Phi)
        B_fast = self.compute_B_fast(Phi)
        # u(t) is mapped slow state target
        u = np.tanh(self.W_c.T @ x_fixed[:self.d_c])
        dydt = (1.0 / self.epsilon) * (A_fast @ y + B_fast @ u)
        return dydt

    def simulate_interval(self, start_step, end_step, n_trials=1):
        trajectories = []
        references = []
        rng = np.random.RandomState(42)
        n_tokens = (end_step - start_step) * int(self.dt_cog / self.dt)
        times = np.linspace(0.0, n_tokens * self.dt, n_tokens)

        for _ in range(n_trials):
            y_traj = np.zeros((n_tokens, self.d_m))
            y_ref_traj = np.zeros((n_tokens, self.d_m))

            y = rng.randn(self.d_m) * 1.5
            # Generate a constant modulation field Phi for this macro-interval
            Phi = rng.randn(self.d_c) * 0.1
            # Generate constant excitation input u
            u = np.ones(self.d_m) * 0.5
            
            A_fast = self.compute_A_fast(Phi)
            B_fast = self.compute_B_fast(Phi)
            
            # The slow invariant manifold is stationary during this macro-interval:
            y_ref = -np.linalg.solve(A_fast, B_fast @ u)

            for k in range(n_tokens):
                # Integrated state updates
                dydt = (1.0 / self.epsilon) * (A_fast @ y + B_fast @ u)
                y = y + dydt * self.dt + self.noise_proc * rng.randn(self.d_m) * np.sqrt(self.dt)
                
                y_traj[k] = y.copy()
                y_ref_traj[k] = y_ref.copy()

            trajectories.append((times, y_traj))
            references.append((times, y_ref_traj))

        return {'trajectories': trajectories, 'reference': references}


def compute_necd(h_model, h_reference):
    if isinstance(h_model, tuple):
        h_model = h_model[1]
    if isinstance(h_reference, tuple):
        h_reference = h_reference[1]
    
    necd = np.zeros(len(h_model))
    for i in range(len(h_model)):
        n1, n2 = np.linalg.norm(h_model[i]), np.linalg.norm(h_reference[i])
        if n1 * n2 > 1e-12:
            sim = np.clip(np.dot(h_model[i], h_reference[i]) / (n1 * n2), -1, 1)
            necd[i] = 1.0 - sim
    
    times = np.linspace(0.0, len(h_model) / 50.0, len(h_model))
    return necd, times


def compute_cpv(log_prob_traj):
    return float(np.var(log_prob_traj))


def _trial(trial_seed, n_tokens, perturb_at_token):
    ts = TokenStream(d_hidden=128, d_semantic=64, seed=trial_seed)
    times, stream = ts.generate(n_tokens, perturb_step=perturb_at_token,
                                perturb_mag=5.0, seed=trial_seed)

    mono = MonolithicTracker(d_hidden=128, seed=trial_seed + 1)
    h_mono, lp_mono = mono.simulate(n_tokens, stream, seed=trial_seed + 1)

    singular_ssm = SingularSSMTracker(d_hidden=128, d_semantic=64, epsilon=0.04, seed=trial_seed + 2)
    h_singular_ssm, lp_singular_ssm = singular_ssm.simulate(n_tokens, stream, seed=trial_seed + 2)

    zoh_r = ZOHTracker(d_hidden=128, d_semantic=64, epsilon=0.04, seed=trial_seed + 3,
                       variant="rigid")
    h_zoh_r, lp_zoh_r = zoh_r.simulate(n_tokens, stream, seed=trial_seed + 3)

    zoh_a = ZOHTracker(d_hidden=128, d_semantic=64, epsilon=0.04, seed=trial_seed + 4,
                       variant="adaptive")
    h_zoh_a, lp_zoh_a = zoh_a.simulate(n_tokens, stream, seed=trial_seed + 4)

    necd_singular_ssm, _ = compute_necd(h_singular_ssm, h_mono)
    necd_rz, _ = compute_necd(h_zoh_r, h_mono)
    necd_az, _ = compute_necd(h_zoh_a, h_mono)

    post = times >= (perturb_at_token / 50.0)
    return {
        'singular_ssm': np.max(necd_singular_ssm[post]) if np.any(post) else 0,
        'zoh_rigid': np.max(necd_rz[post]) if np.any(post) else 0,
        'zoh_adapt': np.max(necd_az[post]) if np.any(post) else 0,
        'cpv_singular_ssm': compute_cpv(lp_singular_ssm[post]),
        'cpv_zoh_r': compute_cpv(lp_zoh_r[post]),
        'cpv_zoh_a': compute_cpv(lp_zoh_a[post]),
        'cpv_mono': compute_cpv(lp_mono[post]),
    }


def run(n_trials=100, n_tokens=500, perturb_at_token=250, seed_base=0):
    h_nd = []; zr_nd = []; za_nd = []
    h_cv = []; zr_cv = []; za_cv = []; m_cv = []

    for t in range(n_trials):
        r = _trial(seed_base + t * 100, n_tokens, perturb_at_token)
        h_nd.append(r['singular_ssm']); zr_nd.append(r['zoh_rigid']); za_nd.append(r['zoh_adapt'])
        h_cv.append(r['cpv_singular_ssm']); zr_cv.append(r['cpv_zoh_r'])
        za_cv.append(r['cpv_zoh_a']); m_cv.append(r['cpv_mono'])

    results = {
        'singular_ssm_necd': float(np.mean(h_nd)), 'singular_ssm_necd_std': float(np.std(h_nd)),
        'zoh_rigid_necd': float(np.mean(zr_nd)), 'zoh_rigid_necd_std': float(np.std(zr_nd)),
        'zoh_adapt_necd': float(np.mean(za_nd)), 'zoh_adapt_necd_std': float(np.std(za_nd)),
        'singular_ssm_cpv': float(np.mean(h_cv)), 'zoh_rigid_cpv': float(np.mean(zr_cv)),
        'zoh_adapt_cpv': float(np.mean(za_cv)), 'mono_cpv': float(np.mean(m_cv)),
        'n': n_trials,
        'singular_ssm_raw_necd': h_nd,
        'zoh_rigid_raw_necd': zr_nd,
        'singular_ssm_raw_cpv': h_cv,
        'zoh_rigid_raw_cpv': zr_cv,
    }
    return results


def run_baseline_comparison(n_trials=100, T_total=200, perturb_time=100.0):
    results = run(n_trials=n_trials, n_tokens=T_total, perturb_at_token=int(perturb_time))

    ts = TokenStream(d_hidden=128, d_semantic=64, seed=42)
    times, stream = ts.generate(T_total, perturb_step=int(perturb_time), seed=42)
    mono = MonolithicTracker(d_hidden=128, seed=42)
    h_mono, _ = mono.simulate(T_total, stream, seed=42)
    singular_ssm = SingularSSMTracker(d_hidden=128, d_semantic=64, epsilon=0.04, seed=42)
    h_singular_ssm, _ = singular_ssm.simulate(T_total, stream, seed=42)

    results['singular_ssm_results'] = {
        'trajectories': [(times, h_singular_ssm)],
        'reference': [(times, h_mono)]
    }
    results['singular_ssm_necd_max'] = results['singular_ssm_necd']
    results['baseline_necd_max'] = results['zoh_rigid_necd']
    results['singular_ssm_cpv'] = results['singular_ssm_cpv']
    results['baseline_cpv'] = results['zoh_rigid_cpv']

    # Calculate real Welch's t-test and Cohen's d from raw trial arrays dynamically
    from scipy import stats
    h_raw = np.array(results['singular_ssm_raw_necd'])
    zr_raw = np.array(results['zoh_rigid_raw_necd'])

    t_stat, p_val = stats.ttest_ind(h_raw, zr_raw, equal_var=False)

    n1, n2 = len(h_raw), len(zr_raw)
    s1, s2 = np.std(h_raw, ddof=1), np.std(zr_raw, ddof=1)
    mean1, mean2 = np.mean(h_raw), np.mean(zr_raw)

    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    num = (s1**2 / n1 + s2**2 / n2) ** 2
    denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    df = num / denom

    results['t_statistic'] = float(t_stat)
    results['df'] = float(df)
    results['p_value'] = float(p_val)
    results['cohens_d'] = float(cohens_d)

    return results


if __name__ == "__main__":
    N = 100
    print("=" * 72)
    print("HONEST SIMULATION: Singular-SSM vs TWO ZOH VARIANTS")
    print("=" * 72)
    print(f"n={N} trials, perturbation at token step 250\n")

    r = run(n_trials=N)

    print(f"{'Model':20s} | {'NECD max':>10s} ± {'std':>8s} | {'CPV':>10s}")
    print("-" * 55)
    print(f"{'Singular-SSM':20s} | {r['singular_ssm_necd']:10.4f} ± {r['singular_ssm_necd_std']:8.4f} | {r['singular_ssm_cpv']:10.2f}")
    print(f"{'ZOH (rigid)':20s} | {r['zoh_rigid_necd']:10.4f} ± {r['zoh_rigid_necd_std']:8.4f} | {r['zoh_rigid_cpv']:10.2f}")
    print(f"{'ZOH (adaptive)':20s} | {r['zoh_adapt_necd']:10.4f} ± {r['zoh_adapt_necd_std']:8.4f} | {r['zoh_adapt_cpv']:10.2f}")
    print(f"{'Monolithic':20s} | {'(ref)':>10s}           | {r['mono_cpv']:10.2f}")
