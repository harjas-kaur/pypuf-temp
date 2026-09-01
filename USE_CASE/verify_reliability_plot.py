
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pypuf.simulation import ArbiterPUF
from pypuf.io import ChallengeResponseSet
from pypuf.metrics import reliability_at_point
import os

# Ensure the script can find the pypuf module
# This adds the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute reliability surface and show interactive 3D plot')
    parser.add_argument('--challenges', '-N', type=int, default=5000,
                        help='Number of challenges to sample (default: 5000). Increase for higher accuracy.')
    args = parser.parse_args()

    # 1. Define PUF parameters
    n_bits = 64
    vdd_nominal = 1.2
    temp_nominal = 25.0
    num_challenges = args.challenges

    # 2. Generate challenges
    challenges = np.random.choice([-1, 1], size=(num_challenges, n_bits))

    # 3. Generate reference CRP set at nominal conditions
    print("Generating reference CRP set...")
    ref_puf = ArbiterPUF(n=n_bits, seed=1, temperature=temp_nominal, vdd=vdd_nominal, noisiness=0.05)
    ref_crps = ChallengeResponseSet(challenges=challenges, responses=ref_puf.eval(challenges))
    print("Reference set created.")

    # 4. Define temperature and voltage sweep ranges (expanded extremes)
    # Temperatures from -40°C to 150°C and Vdd from 0.1V to 5.0V
    temp_sweep = np.arange(-40.0, 151.0, 5.0)
    vdd_sweep = np.round(np.arange(0.1, 5.01, 0.05), 2)

    reliability_grid = np.zeros((len(temp_sweep), len(vdd_sweep)))

    # 5. Compute reliability over the full temperature/voltage grid
    print("Computing reliability surface...")
    for i, temp in enumerate(temp_sweep):
        for j, vdd in enumerate(vdd_sweep):
            puf_at_point = ArbiterPUF(n=n_bits, seed=1, temperature=temp, vdd=vdd, noisiness=0.05)
            reliability_grid[i, j] = np.mean(reliability_at_point(puf_at_point, ref_crps))
        print(f"  Temp={temp:.1f}°C complete")

    # No forced crosshair lines: show only simulation-derived reliability values

    # 6. Create meshgrid for plotting
    VDD, TEMP = np.meshgrid(vdd_sweep, temp_sweep)

    # 7. Plot the 3D reliability surface
    print("Plotting the 3D reliability surface...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Use inverted colormap to match the reference (colors reversed)
    surf = ax.plot_surface(VDD, TEMP, reliability_grid, cmap='viridis_r', edgecolor='k', linewidth=0.2, antialiased=True)
    ax.set_title(f'Reliability Surface vs. Temperature and Vdd (n={n_bits})')
    ax.set_xlabel('Vdd (V)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_zlabel('Reliability')
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Reliability')

    # No overlay crosshairs; view the raw simulated surface

    surface_filename = 'reliability_surface_3d.png'
    plt.savefig(surface_filename)
    print(f"3D surface plot saved to {surface_filename}")

    # Show interactive matplotlib window so you can rotate/zoom the surface
    print("Opening interactive 3D plot window (close the window to continue)...")
    plt.show()

    # 8. Optionally save a second 2D contour projection for reference
    print("Plotting the projection contour heatmap (inverted colormap)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Inverted colormap to match the reference image style
    contour = ax.contourf(VDD, TEMP, reliability_grid, levels=50, cmap='viridis_r')
    ax.set_title(f'Reliability Projection vs. Temperature and Vdd (n={n_bits})')
    ax.set_xlabel('Vdd (V)')
    ax.set_ylabel('Temperature (°C)')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Reliability')

    heatmap_filename = 'reliability_projection_heatmap.png'
    plt.savefig(heatmap_filename)
    print(f"Projection heatmap saved to {heatmap_filename}")
    plt.close()

if __name__ == "__main__":
    main()
