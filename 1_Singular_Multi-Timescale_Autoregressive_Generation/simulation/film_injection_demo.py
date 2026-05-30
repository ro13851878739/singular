import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import os


class FiLMInjectionCircuit:
    """
    Tensor-Aligned Feature-wise Linear Modulation (FiLM) Injection Circuit
    
    Implements Section III equations:
    
    1. Projection: gamma_hat = W_gamma * Phi + b_gamma
                   delta_hat = W_delta * Phi + b_delta
    
    2. Spatial Broadcasting: gamma = M_broad(gamma_hat)  (B x N x D)
                             delta = M_broad(delta_hat)  (B x N x D)
    
    3. Modulation: H_tilde = (1 + tanh(gamma)) ⊙ H_f + delta
    
    This circuit injects slow-rate semantic context into fast token-level processing
    without evaluating the full cognitive backbone at every step.
    """

    def __init__(self, D=256, d_c=64, N=512):
        """
        Args:
            D: Channel dimension (hidden size)
            d_c: Conditioning field dimension (cognitive state dim)
            N: Sequence length (number of tokens)
        """
        self.D = D
        self.d_c = d_c
        self.N = N

        # Learnable projection parameters (simulated as random for demo)
        np.random.seed(42)
        self.W_gamma = np.random.randn(D, d_c) * 0.1  # Scaling projection
        self.b_gamma = np.zeros(D)
        self.W_delta = np.random.randn(D, d_c) * 0.1  # Shift projection
        self.b_delta = np.zeros(D)

    def project(self, Phi):
        """
        Compute time-varying scaling and shift vectors from conditioning field
        
        Eq. (4): gamma_hat = W_gamma * Phi + b_gamma
                 delta_hat = W_delta * Phi + b_delta
        """
        gamma_hat = self.W_gamma @ Phi + self.b_gamma
        delta_hat = self.W_delta @ Phi + self.b_delta
        return gamma_hat, delta_hat

    def spatial_broadcast(self, vector, B=1):
        """
        Spatial Broadcasting Operator M_broad
        
        Replicates channel vector across sequence dimension:
        Input: (D,) -> Output: (B, N, D)
        
        Eq. (5): gamma = M_broad(gamma_hat) ∈ R^{B×N×D}
        """
        if len(vector.shape) == 1:
            broadcasted = np.tile(vector, (B, self.N, 1))
        else:
            broadcasted = np.tile(vector[:, np.newaxis, :], (1, self.N, 1))
        return broadcasted

    def modulate(self, H_f, Phi, B=1):
        """
        Apply FiLM modulation to hidden activations
        
        Eq. (6): H_tilde = (1 + tanh(gamma)) ⊙ H_f + delta
        
        This is the core injection mechanism that reshapes surface manifold
        gradients using slow-rate semantic context.
        
        Args:
            H_f: Input hidden activations (B, N, D) - from Transformer layer l
            Phi: Conditioning field from T1 cognitive layer (d_c,)
            B: Batch size
            
        Returns:
            H_tilde: Modulated activations (B, N, D)
        """
        # Step 1: Project conditioning field to scaling/shift
        gamma_hat, delta_hat = self.project(Phi)

        # Step 2: Spatial broadcasting to match activation tensor shape
        gamma = self.spatial_broadcast(gamma_hat, B)
        delta = self.spatial_broadcast(delta_hat, B)

        # Step 3: Element-wise modulation with tanh nonlinearity
        # tanh ensures scaling factor stays in [-1, 1] range (stable)
        scale_factor = 1.0 + np.tanh(gamma)  # Range: [0, 2]

        # Feature-wise modulation (channel-wise multiplication + addition)
        H_tilde = scale_factor * H_f + delta

        return H_tilde, {
            'gamma': gamma,
            'delta': delta,
            'scale_factor': scale_factor,
            'gamma_hat': gamma_hat,
            'delta_hat': delta_hat
        }

    def visualize_modulation_process(self, save_path=None):
        """
        Create comprehensive visualization of the FiLM injection process
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Generate sample data
        B, N, D = 1, self.N, self.D
        H_f = np.random.randn(B, N, D) * 0.5  # Input activations
        Phi = np.random.randn(self.d_c) * 1.0   # Conditioning field

        # Apply modulation
        H_tilde, intermediates = self.modulate(H_f, Phi, B)

        # Plot 1: Input hidden state distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(H_f.flatten(), bins=50, alpha=0.7, color='blue', label='Input $H_f$')
        ax1.set_title('Distribution of Input Activations\n$H_f^{(l)}(t)$', fontsize=11)
        ax1.set_xlabel('Activation Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Conditioning field
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(range(self.d_c), Phi, alpha=0.7, color='green', label='$\\Phi(t, H_t)$')
        ax2.set_title(f'Conditioning Field\n(Dimension: {self.d_c})', fontsize=11)
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('$\\Phi$ Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scaling factor (gamma) statistics
        ax3 = fig.add_subplot(gs[0, 2])
        scale_flat = intermediates['scale_factor'].flatten()
        ax3.hist(scale_flat, bins=50, alpha=0.7, color='orange', label='Scale Factor')
        ax3.axvline(x=1.0, color='red', linestyle='--', label='Identity (no scaling)')
        ax3.set_title('Scale Factor Distribution\n$(1 + \\tanh(\\gamma))$', fontsize=11)
        ax3.set_xlabel('Scale Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.text(0.95, 0.95, f'Mean: {np.mean(scale_flat):.3f}\nStd: {np.std(scale_flat):.3f}',
                transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 4: Output modulated state
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(H_tilde.flatten(), bins=50, alpha=0.7, color='red', label='Output $\\tilde{H}_f$')
        ax4.set_title('Distribution of Modulated Activations\n$\\tilde{H}_f^{(l)}(t)$', fontsize=11)
        ax4.set_xlabel('Activation Value')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Modulation effect on a single token position
        ax5 = fig.add_subplot(gs[1, 1])
        token_idx = N // 2  # Middle token
        ax5.plot(range(D), H_f[0, token_idx, :], 'b-', alpha=0.7, label='Input $H_f$')
        ax5.plot(range(D), H_tilde[0, token_idx, :], 'r-', alpha=0.7, label='Modulated $\\tilde{H}_f$')
        ax5.fill_between(range(D), H_f[0, token_idx, :], H_tilde[0, token_idx, :],
                        alpha=0.3, color='purple', label='Modulation Effect')
        ax5.set_title(f'Modulation at Token Position {token_idx}', fontsize=11)
        ax5.set_xlabel('Channel Index')
        ax5.set_ylabel('Activation Value')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Shift term (delta) distribution
        ax6 = fig.add_subplot(gs[1, 2])
        delta_flat = intermediates['delta'].flatten()
        ax6.hist(delta_flat, bins=50, alpha=0.7, color='purple', label='Shift $\\delta$')
        ax6.axvline(x=0.0, color='black', linestyle='--', label='Zero shift')
        ax6.set_title('Shift Term Distribution\n$\\delta^{(l)}(t)$', fontsize=11)
        ax6.set_xlabel('Shift Value')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Plot 7: Spatial pattern visualization (heatmap)
        ax7 = fig.add_subplot(gs[2, 0])
        spatial_pattern = intermediates['scale_factor'][0, :, :min(32, D)].T  # Subsample for visibility
        im = ax7.imshow(spatial_pattern, aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=2)
        ax7.set_title('Spatial Scale Pattern\n(Token × Channel subsample)', fontsize=11)
        ax7.set_xlabel('Token Position (subsampled)')
        ax7.set_ylabel('Channel Index')
        plt.colorbar(im, ax=ax7, label='Scale Factor')

        # Plot 8: Information flow diagram
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')

        flow_text = """
        ╔══════════════════════════════════════╗
        ║     FiLM Injection Circuit Flow       ║
        ╠══════════════════════════════════════╣
        ║                                      ║
        ║  Φ(t,H_t) ──┐                        ║
        ║              ▼                        ║
        ║     ┌───────────────┐                ║
        ║     │  W_γ, W_δ     │  Projection    ║
        ║     │  b_γ, b_δ     │  (Eq. 4)       ║
        ║     └───────┬───────┘                ║
        ║             │                         ║
        ║             ▼                         ║
        ║     ┌───────────────┐                ║
        ║     │  M_broad      │  Broadcast     ║
        ║     │  (Spatial)    │  (Eq. 5)       ║
        ║     └───────┬───────┘                ║
        ║             │                         ║
        ║    γ, δ ────┤                         ║
        ║             │                         ║
        ║  H_f ───────┤───⊙── (+) ──→ H̃_f      ║
        ║ (Input)     │  (1+tanhγ)   δ          ║
        ║             │              (Eq. 6)    ║
        ╚══════════════════════════════════════╝
        """
        ax8.text(0.05, 0.95, flow_text, transform=ax8.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Plot 9: Summary statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_stats = f"""
        ╔═══════════════════════════════╗
        ║   FiLM Circuit Statistics      ║
        ╠═══════════════════════════════╣
        ║                                ║
        ║  Tensor Dimensions:            ║
        ║    • B (Batch): {B:>4}           ║
        ║    • N (Tokens): {N:>4}          ║
        ║    • D (Channels): {D:>4}         ║
        ║    • d_c (Conditioning): {self.d_c:>4}  ║
        ║                                ║
        ║  Modulation Effects:           ║
        ║    • Scale Mean: {np.mean(scale_flat):>6.3f}       ║
        ║    • Scale Std:  {np.std(scale_flat):>6.3f}       ║
        ║    • Shift Mean: {np.mean(delta_flat):>7.3f}      ║
        ║    • Shift Std:  {np.std(delta_flat):>7.3f}      ║
        ║                                ║
        ║  Stability Check:              ║
        ║    • Scale in [0,2]: {'✓' if np.all((scale_flat>=0)&(scale_flat<=2)) else '✗'}        ║
        ║    • No NaN/Inf:  {'✓' if not (np.any(np.isnan(H_tilde)) or np.any(np.isinf(H_tilde))) else '✗'}          ║
        ╚═════════════════════════════════╝
        """
        ax9.text(0.05, 0.95, summary_stats, transform=ax9.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.suptitle('Tensor-Aligned FiLM Injection Circuit Visualization\n'
                    '(Section III: Multi-Rate Context Injection Mechanism)',
                    fontsize=14, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved to: {save_path}")

        plt.tight_layout()
        plt.close(fig)

        return H_tilde, intermediates


def demonstrate_temporal_modulation():
    """
    Demonstrate how FiLM modulation evolves over time as conditioning field changes
    Shows the continuous-time nature of the cross-frequency interface
    """
    print("=" * 70)
    print("Temporal FiLM Modulation Demonstration")
    print("=" * 70)
    print("\nSimulating how conditioning field Φ(t) evolves over cognitive intervals...")
    print("and how this affects token-level modulations at 50Hz.\n")

    film = FiLMInjectionCircuit(D=128, d_c=64, N=256)

    # Simulate over multiple cognitive intervals
    n_intervals = 3
    tokens_per_interval = int(0.5 / (1.0/50.0))  # 25 tokens per 500ms interval at 50Hz

    all_H_tilde = []
    all_Phi = []
    all_times = []

    t_global = 0.0
    dt_token = 1.0 / 50.0

    for interval in range(n_intervals):
        # Cognitive update at start of interval
        Phi_base = np.random.randn(64) * (1.0 + interval * 0.5)  # Increasing magnitude

        for step in range(tokens_per_interval):
            t_in_interval = step * dt_token
            
            # Exponential attenuation of conditioning field within interval
            Phi_t = Phi_base * np.exp(-2.0 * t_in_interval)  # Decaying conditioning

            # Generate random input activations
            H_f = np.random.randn(1, 256, 128) * 0.5

            # Apply modulation
            H_tilde, _ = film.modulate(H_f, Phi_t, B=1)

            all_H_tilde.append(H_tilde.copy())
            all_Phi.append(Phi_t.copy())
            all_times.append(t_global)

            t_global += dt_token

    # Visualize temporal evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Conditioning field evolution
    ax1 = axes[0, 0]
    Phi_array = np.array(all_Phi)
    im1 = ax1.imshow(Phi_array.T, aspect='auto', cmap='viridis')
    ax1.set_title('Conditioning Field Evolution $\\Phi(t)$\n(Exponential Decay per Interval)', fontsize=11)
    ax1.set_xlabel('Time Step (50Hz)')
    ax1.set_ylabel('Channel Index')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Interval Boundary')
    for i in range(1, n_intervals):
        ax1.axhline(y=i*tokens_per_interval, color='red', linestyle='--', alpha=0.3)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Plot 2: Modulation intensity over time
    ax2 = axes[0, 1]
    modulation_norms = [np.linalg.norm(H.flatten()) for H in all_H_tilde]
    ax2.plot(all_times, modulation_norms, 'b-', linewidth=1.5, label='$||\\tilde{H}_f(t)||$')
    ax2.set_title('Modulation Intensity Over Time', fontsize=11)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('L2 Norm of Modulated State')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark cognitive update times
    for i in range(n_intervals):
        ax2.axvline(x=i*0.5, color='red', linestyle='--', alpha=0.5, label=f'Cognitive Update {i+1}' if i==0 else '')

    # Plot 3: Sample channel trajectory
    ax3 = axes[1, 0]
    sample_channel = 64
    sample_token = 128
    channel_values = [H[0, sample_token, sample_channel] for H in all_H_tilde]
    ax3.plot(all_times, channel_values, 'g-', linewidth=1.5)
    ax3.set_title(f'Single Channel Trajectory\n(Token={sample_token}, Channel={sample_channel})', fontsize=11)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Activation Value')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Statistical summary per interval
    ax4 = axes[1, 1]
    interval_means = []
    interval_stds = []
    for i in range(n_intervals):
        start = i * tokens_per_interval
        end = (i+1) * tokens_per_interval
        interval_data = [np.linalg.norm(H.flatten()) for H in all_H_tilde[start:end]]
        interval_means.append(np.mean(interval_data))
        interval_stds.append(np.std(interval_data))

    x_pos = np.arange(n_intervals)
    width = 0.35
    bars1 = ax4.bar(x_pos - width/2, interval_means, width, label='Mean ||H̃||', color='steelblue', yerr=interval_stds, capsize=5)
    ax4.set_title('Per-Interval Modulation Statistics', fontsize=11)
    ax4.set_xlabel('Cognitive Interval')
    ax4.set_ylabel('L2 Norm')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Interval {i+1}\n({i*0.5:.1f}-{(i+1)*0.5:.1f}s)' for i in range(n_intervals)])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Temporal Dynamics of FiLM Injection Circuit\n'
                '(Continuous-Time Cross-Frequency Interface at 50Hz)',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures/film_temporal_dynamics.png")),
               dpi=300, bbox_inches='tight')
    print("✅ Temporal dynamics figure saved!")

    plt.close('all')

    return all_H_tilde, all_Phi, all_times


if __name__ == "__main__":
    print("#" * 70)
    print("# FiLM Injection Circuit Demonstration")
    print("# Section III: Tensor-Aligned Feature-wise Linear Modulation")
    print("#" * 70)
    print()

    # Static visualization
    film = FiLMInjectionCircuit(D=256, d_c=64, N=512)
    H_tilde, intermediates = film.visualize_modulation_process(
        save_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures/film_circuit_visualization.png"))
    )

    # Temporal dynamics
    demonstrate_temporal_modulation()

    print("\n✅ FiLM demonstration complete!")
