
import numpy as np
import torch
import torch.nn as nn

# Optional imports - will fail gracefully if not installed
try:
    import gudhi
except ImportError:
    gudhi = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class TrajectoryHook:
    """
    A PyTorch hook to capture and store the hidden state trajectory of a recurrent block.
    
    Since the unrolling loop in Huginn (RavenForCausalLM) executes the core block sequentially,
    registering the hook on the last block in the recurrent block container (transformer.core_block[-1])
    ensures that we capture the hidden state exactly once at the end of each recurrent step.
    """
    def __init__(self, target_module: nn.Module):
        self.target_module = target_module
        self.trajectory: list[torch.Tensor] = []
        self.hook_handle = target_module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input_t, output_t):
        if isinstance(output_t, tuple):
            state = output_t[0]
        else:
            state = output_t

        # Detach and move to CPU to avoid memory leakage/GPU memory bloat
        self.trajectory.append(state.detach().cpu())

    def clear(self):
        self.trajectory = []

    def remove(self):
        self.hook_handle.remove()

    def get_trajectory(self) -> torch.Tensor:
        """
        Returns the trajectory as a tensor of shape:
        [num_steps, batch_size, sequence_length, hidden_dim]
        """
        if not self.trajectory:
            return torch.empty(0)
        return torch.stack(self.trajectory, dim=0)


def extract_token_trajectory(trajectory: torch.Tensor, token_idx: int = -1, batch_idx: int = 0) -> np.ndarray:
    """
    Extracts the trajectory of a single token from the stacked trajectory tensor.
    
    Args:
        trajectory: Tensor of shape [num_steps, batch_size, seq_len, hidden_dim]
        token_idx: Index of the token to track (default: -1, the last prompt/generated token)
        batch_idx: Batch index (default: 0)
        
    Returns:
        np.ndarray of shape [num_steps, hidden_dim]
    """
    token_traj = trajectory[:, batch_idx, token_idx, :]
    return token_traj.numpy()


def compute_step_displacements(trajectory_np: np.ndarray) -> np.ndarray:
    """
    Computes the step-to-step displacement (distance) between consecutive recurrent states.
    
    Args:
        trajectory_np: Array of shape [num_steps, hidden_dim]
        
    Returns:
        np.ndarray of shape [num_steps - 1] containing L2 distances
    """
    diffs = np.diff(trajectory_np, axis=0)
    displacements = np.linalg.norm(diffs, axis=1)
    return displacements


def compute_finite_time_lyapunov(trajectory_np: np.ndarray) -> float:
    """
    Computes a simplified estimate of the largest finite-time Lyapunov exponent (FTLE) 
    by checking the divergence of close-by points along the path.
    
    Args:
        trajectory_np: Array of shape [num_steps, hidden_dim]
        
    Returns:
        float: Estimated largest Lyapunov exponent.
    """
    num_steps = len(trajectory_np)
    if num_steps < 3:
        return 0.0

    diffs = np.diff(trajectory_np, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    norms = np.clip(norms, 1e-12, None)

    ratios = norms[1:] / norms[:-1]
    lyapunovs = np.log(ratios)

    return float(np.mean(lyapunovs))


def compute_winding_number_2d(trajectory_np: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Projects the trajectory to its top 2 principal components
    and computes the winding number around the centroid of the trajectory.
    
    Args:
        trajectory_np: Array of shape [num_steps, hidden_dim]
        
    Returns:
        Tuple:
            - float: Winding number (total angle accumulated divided by 2*pi)
            - np.ndarray: Projected 2D coordinates [num_steps, 2]
            - np.ndarray: Accumulated angles over steps [num_steps]
    """
    if PCA is None:
        raise ImportError("sklearn is required for PCA projection. Please install scikit-learn.")

    num_steps = len(trajectory_np)
    if num_steps < 3:
        return 0.0, np.zeros((num_steps, 2)), np.zeros(num_steps)

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(trajectory_np)

    centroid = np.mean(coords_2d, axis=0)
    centered = coords_2d - centroid

    angles = np.arctan2(centered[:, 1], centered[:, 0])

    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

    cum_angles = np.zeros(num_steps)
    cum_angles[1:] = np.cumsum(angle_diffs)

    total_angle = cum_angles[-1]
    winding_number = total_angle / (2 * np.pi)

    return float(winding_number), coords_2d, cum_angles


def compute_persistent_homology(trajectory_np: np.ndarray, max_dimension: int = 1) -> dict[str, list]:
    """
    Computes persistent homology of the trajectory point cloud in high-dimensional space
    using Gudhi's Vietoris-Rips complex.
    
    Args:
        trajectory_np: Array of shape [num_steps, hidden_dim]
        max_dimension: Maximum topological dimension to compute (0 = components, 1 = loops)
        
    Returns:
        Dict containing:
            - 'betti_numbers': List of Betti numbers at the end of filtration
            - 'persistence_intervals': Dict mapping dimension to lists of (birth, death) tuples
    """
    if gudhi is None:
        return {"error": "gudhi package is not installed. Skipping PH computation."}

    rips = gudhi.RipsComplex(points=trajectory_np)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)

    persistence = simplex_tree.persistence()
    simplex_tree.compute_persistence()
    betti = simplex_tree.betti_numbers()

    intervals = {d: [] for d in range(max_dimension + 1)}
    for dim, (birth, death) in persistence:
        if dim <= max_dimension:
            intervals[dim].append((birth, death))

    return {
        "betti_numbers": betti,
        "persistence_intervals": intervals
    }


def plot_trajectory_analysis(
    displacements: np.ndarray,
    coords_2d: np.ndarray,
    cum_angles: np.ndarray,
    winding_no: float,
    lyapunov: float,
    save_path: str = "trajectory_analysis.png"
):
    """
    Generates and saves a three-panel visualization figure for publication/presentation slides.
    """
    if plt is None:
        print("matplotlib not installed. Skipping figure generation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 2D PCA Trajectory Flow
    ax = axes[0]
    num_steps = len(coords_2d)
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=range(num_steps), cmap="viridis", edgecolor="k", zorder=3)
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color="gray", linestyle="--", alpha=0.6, zorder=2)

    # Add step flow indicators (arrows)
    for i in range(0, num_steps - 1, max(1, num_steps // 8)):
        dx = coords_2d[i+1, 0] - coords_2d[i, 0]
        dy = coords_2d[i+1, 1] - coords_2d[i, 1]
        ax.annotate("", xy=(coords_2d[i+1, 0], coords_2d[i+1, 1]), xytext=(coords_2d[i, 0], coords_2d[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=0, shrinkB=0))

    # Centroid
    centroid = np.mean(coords_2d, axis=0)
    ax.scatter(centroid[0], centroid[1], color="red", marker="x", s=100, label="Centroid", zorder=4)

    ax.set_title("PCA Projection (2D) of Latent Path", fontsize=12)
    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Recurrence Step $t$")

    # 2. Step-to-step Displacements
    ax = axes[1]
    ax.plot(range(1, len(displacements) + 1), displacements, marker="o", color="#4f46e5", lw=2)
    ax.set_title("Step-to-Step Displacement $||h_{t+1} - h_t||_2$", fontsize=12)
    ax.set_xlabel("Step $t$", fontsize=10)
    ax.set_ylabel("Distance", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Add regime info label
    regime = "Settling" if displacements[-1] < 0.1 * displacements[0] else ("Looping" if np.std(displacements[-5:]) < 0.05 * np.mean(displacements) else "Drifting/Unstable")
    ax.text(0.05, 0.95, f"Regime: {regime}\nLyapunov: {lyapunov:.4f}", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # 3. Cumulative Angle (Winding Progression)
    ax = axes[2]
    ax.plot(range(num_steps), cum_angles, marker="s", color="#06b6d4", lw=2)
    ax.set_title("Accumulated Trajectory Angle (Winding)", fontsize=12)
    ax.set_xlabel("Step $t$", fontsize=10)
    ax.set_ylabel("Accumulated Angle (radians)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    for j in range(-int(abs(winding_no))-1, int(abs(winding_no))+2):
        if j != 0:
            ax.axhline(j * 2 * np.pi, color="red", linestyle=":", alpha=0.3)
            ax.text(0.9, j * 2 * np.pi, f"${j}\\times 2\\pi$", transform=ax.get_yaxis_transform(), fontsize=8, color="red")

    ax.text(0.05, 0.95, f"Winding Number: {winding_no:.4f} turns", transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved trajectory analysis figure to {save_path}")
    plt.close()


def instrument_model_run(
    model_name: str = "tomg-group-umd/huginn-0125",
    prompt: str = "Solve the following puzzle: 5 3 . . 7 . . . . \n ...",
    num_steps: int = 32,
    token_idx: int = -1,
    figure_path: str = "trajectory_analysis.png"
):
    """
    Example execution flow showing how to load the model, attach the hook,
    run inference with specified num_steps, and compute/plot the geometric metrics.
    """
    print(f"Loading model: {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Locate the target block to hook.
    # Huginn (RavenForCausalLM) unrolls over model.transformer.core_block (a ModuleList of SandwichBlocks).
    # We hook the last SandwichBlock inside the core_block to intercept the hidden state
    # exactly at the end of each recurrent step.
    target_module = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "core_block"):
        target_module = model.transformer.core_block[-1]
        print("Success: Located model.transformer.core_block[-1] directly.")
    else:
        # Fallback recursive lookup
        for name, module in model.named_modules():
            if "transformer.core_block" in name and isinstance(module, nn.ModuleList) and len(module) > 0:
                target_module = module[-1]
                print(f"Success: Located last block of recurrent container: {name}[-1]")
                break

    if target_module is None:
        # General backup search
        for name, module in model.named_modules():
            if "recurrent" in name.lower() and isinstance(module, nn.ModuleList) and len(module) > 0:
                target_module = module[-1]
                print(f"Fallback: Hooked last block of recurrent module: {name}[-1]")
                break

    if target_module is None:
        raise ValueError("Could not automatically locate the recurrent core block. Please configure target manually.")

    hook = TrajectoryHook(target_module)
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"Running inference with num_steps={num_steps}...")
    with torch.no_grad():
        # Pass num_steps as kwargs so it propagates to iterate_forward
        outputs = model(**inputs, num_steps=num_steps)

    raw_traj = hook.get_trajectory()
    token_traj = extract_token_trajectory(raw_traj, token_idx=token_idx)

    displacements = compute_step_displacements(token_traj)
    lyapunov = compute_finite_time_lyapunov(token_traj)

    print("\n--- Geometric Metrics ---")
    print(f"Final step displacement: {displacements[-1] if len(displacements) > 0 else 'N/A'}")
    print(f"Estimated Finite-Time Lyapunov Exponent: {lyapunov:.6f}")

    try:
        winding_no, coords_2d, cum_angles = compute_winding_number_2d(token_traj)
        print(f"Estimated 2D Winding Number: {winding_no:.6f} turns")
        if plt is not None:
            plot_trajectory_analysis(displacements, coords_2d, cum_angles, winding_no, lyapunov, save_path=figure_path)
    except Exception as e:
        print(f"Could not compute winding number or plot: {e}")

    ph_metrics = compute_persistent_homology(token_traj)
    if "error" not in ph_metrics:
        print(f"Betti numbers (B0, B1): {ph_metrics['betti_numbers']}")
    else:
        print(ph_metrics["error"])

    hook.remove()
    return token_traj
