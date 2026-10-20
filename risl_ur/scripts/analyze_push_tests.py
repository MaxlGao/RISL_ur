import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_friction_trials(csv_file, t_start=1.0, t_end=5.0):
    """
    Process friction trial CSV files to extract velocity estimates.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    t_start : float
        Start time for clipping each trial (seconds)
    t_end : float
        End time for clipping each trial (seconds)
    
    Returns:
    --------
    results : pd.DataFrame
        DataFrame with trial results including velocities
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Identify trial boundaries (when Force changes)
    df['trial'] = (df['Force'] != df['Force'].shift()).cumsum()
    
    # Reset time within each trial
    df['trial_time'] = df.groupby('trial')['time'].transform(lambda x: x - x.iloc[0])
    
    results = []
    
    for trial_id, trial_data in df.groupby('trial'):
        Force = trial_data['Force'].iloc[0]
        
        # Clip data to the specified time interval
        clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                             (trial_data['trial_time'] <= t_end)].copy()
        
        if len(clipped) < 2:
            print(f"Warning: Trial {trial_id} (Force={Force}) has insufficient data in interval [{t_start}, {t_end}]")
            continue
        
        # Calculate velocities using finite differences
        dt = np.diff(clipped['trial_time'])
        vx = np.diff(clipped['x']) / dt
        vy = np.diff(clipped['y']) / dt
        tz = np.diff(clipped['Rz']) / dt # Degrees/sec
        
        # Calculate speed
        speed = np.sqrt(vx**2 + vy**2)
        
        # Statistics for the trial
        result = {
            'trial': trial_id,
            'Force': Force,
            'n_points': len(clipped),
            'duration': clipped['trial_time'].iloc[-1] - clipped['trial_time'].iloc[0],
            'vx_mean': np.mean(vx),
            'vx_std': np.std(vx),
            'vy_mean': np.mean(vy),
            'vy_std': np.std(vy),
            'tz_mean': np.mean(tz),
            'tz_std': np.std(tz),
            'speed_mean': np.mean(speed),
            'speed_std': np.std(speed),
            'x_displacement': clipped['x'].iloc[-1] - clipped['x'].iloc[0],
            'y_displacement': clipped['y'].iloc[-1] - clipped['y'].iloc[0],
            'vx_all': vx,  # Store all velocity points
            'vy_all': vy,
            'tz_all': tz,
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def fit_velocity_force_relationship(results):
    """
    Fit a linear relationship: vy = a*Force + b
    
    Parameters:
    -----------
    results : pd.DataFrame
        Results DataFrame from process_friction_trials
    
    Returns:
    --------
    a : float
        Slope of the linear fit
    b : float
        Intercept of the linear fit
    r_squared : float
        R-squared value of the fit
    """
    # Extract Force and vy_mean
    Force = results['Force'].values
    vy = results['vy_mean'].values
    
    # Perform linear regression: vy = a*Force + b
    coeffs = np.polyfit(Force, vy, 1)
    a, b = coeffs[0], coeffs[1]
    
    # Calculate R-squared
    vy_pred = a * Force + b
    ss_res = np.sum((vy - vy_pred)**2)
    ss_tot = np.sum((vy - np.mean(vy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared

def fit_twist_force_relationship(results):
    """
    Fit a linear relationship: tz = a*Force + b
    
    Parameters:
    -----------
    results : pd.DataFrame
        Results DataFrame from process_friction_trials
    
    Returns:
    --------
    a : float
        Slope of the linear fit
    b : float
        Intercept of the linear fit
    r_squared : float
        R-squared value of the fit
    """
    # Extract Force and tz_mean
    Force = results['Force'].values
    tz = results['tz_mean'].values
    
    # Perform linear regression: tz = a*Force + b
    coeffs = np.polyfit(Force, tz, 1)
    a, b = coeffs[0], coeffs[1]
    
    # Calculate R-squared
    tz_pred = a * Force + b
    ss_res = np.sum((tz - tz_pred)**2)
    ss_tot = np.sum((tz - np.mean(tz))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared

def plot_velocity_force_fit(results, a, b, r_squared):
    """
    Plot the velocity-force relationship with violin plots showing distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    Force = results['Force'].values
    vy = results['vy_mean'].values
    vy_std = results['vy_std'].values
    
    # Create violin plots for each force level
    positions = Force
    violin_data = [results.iloc[i]['vy_all'] for i in range(len(results))]
    
    parts = ax.violinplot(violin_data, positions=positions, widths=0.3, 
                          showmeans=True, showmedians=True)
    
    # Customize violin colors
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
    
    # Plot mean points with error bars
    ax.errorbar(Force, vy, yerr=vy_std, fmt='o', markersize=8, 
                capsize=5, label='Mean ± std', alpha=0.8, color='darkblue', zorder=5)
    
    # Plot fitted line
    Force_fit = np.linspace(Force.min() - 0.5, Force.max() + 0.5, 100)
    vy_fit = a * Force_fit + b
    ax.plot(Force_fit, vy_fit, 'r-', linewidth=2, 
            label=f'Fit: vy = {a:.6f}·Force + {b:.6f}')
    
    ax.set_xlabel('Applied Force (N)', fontsize=12)
    ax.set_ylabel('Velocity vy (m/s)', fontsize=12)
    ax.set_title(f'Linear Velocity-Force Relationship (R² = {r_squared:.6f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with fit parameters
    textstr = f'a = {a:.6f} m/(s·N)\nb = {b:.6f} m/s\nR² = {r_squared:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_twist_force_fit(results, a, b, r_squared):
    """
    Plot the twist-force relationship with violin plots showing distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    Force = results['Force'].values
    tz = results['tz_mean'].values
    tz_std = results['tz_std'].values
    
    # Create violin plots for each force level
    positions = Force
    violin_data = [results.iloc[i]['tz_all'] for i in range(len(results))]
    
    parts = ax.violinplot(violin_data, positions=positions, widths=0.3, 
                          showmeans=True, showmedians=True)
    
    # Customize violin colors
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.6)
    
    # Plot mean points with error bars
    ax.errorbar(Force, tz, yerr=tz_std, fmt='o', markersize=8, 
                capsize=5, label='Mean ± std', alpha=0.8, color='darkred', zorder=5)
    
    # Plot fitted line
    Force_fit = np.linspace(Force.min() - 0.5, Force.max() + 0.5, 100)
    tz_fit = a * Force_fit + b
    ax.plot(Force_fit, tz_fit, 'r-', linewidth=2, 
            label=f'Fit: tz = {a:.6f}·Force + {b:.6f}')
    
    ax.set_xlabel('Applied Torque (Nm)', fontsize=12)
    ax.set_ylabel('Angular Velocity tz (deg/s)', fontsize=12)
    ax.set_title(f'Rotational Velocity-Force Relationship (R² = {r_squared:.6f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with fit parameters
    textstr = f'a = {a:.6f} deg/(s·Nm)\nb = {b:.6f} deg/s\nR² = {r_squared:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def create_combined_figure(csv_file, t_start=1.0, t_end=5.0):
    """
    Create a combined figure with trajectory and velocity-force relationship as subplots.
    """
    is_rotational = "rotational" in csv_file
    results = process_friction_trials(csv_file, t_start, t_end)
    
    # Read data for trajectory plots
    df = pd.read_csv(csv_file)
    df['trial'] = (df['Force'] != df['Force'].shift()).cumsum()
    df['trial_time'] = df.groupby('trial')['time'].transform(lambda x: x - x.iloc[0])
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Plot 1: Trajectories in x-y space
    for trial_id, trial_data in df.groupby('trial'):
        Force = trial_data['Force'].iloc[0]
        clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                             (trial_data['trial_time'] <= t_end)]
        
        if len(clipped) > 0:
            ax1.plot(clipped['x'], clipped['y'], 
                    label=f'Force={Force}', marker='o', markersize=2, alpha=0.7)
    
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('X-Y Trajectories')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    if is_rotational:
        ax1.set_title('X-Y Trajectories, Including Orientation')
        # For rotational: add rotation arrows
        for trial_id, trial_data in df.groupby('trial'):
            Force = trial_data['Force'].iloc[0]
            clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                                 (trial_data['trial_time'] <= t_end)]
            
            if len(clipped) > 5:
                step = max(1, len(clipped) // 10)
                for i in range(0, len(clipped) - step, step):
                    x1, y1 = clipped['x'].iloc[i], clipped['y'].iloc[i]
                    dx = np.cos(np.deg2rad(clipped['Rz'].iloc[i]))*0.0005
                    dy = np.sin(np.deg2rad(clipped['Rz'].iloc[i]))*0.0005
                    ax1.arrow(x1, y1, dx, dy, head_width=0.0002, head_length=0.0002, width=0.00005,
                                 fc='black', ec='black', alpha=0.5)
        ax1.axis('equal')

    # Plot 2: Position vs time (normalized)
    for trial_id, trial_data in df.groupby('trial'):
        Force = trial_data['Force'].iloc[0]
        clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                             (trial_data['trial_time'] <= t_end)].copy()
        
        if len(clipped) > 0:
            if is_rotational:
                position = clipped['Rz'] - clipped['Rz'].iloc[0]
                ylabel = 'Rz rotation (deg)'
            else:
                position = clipped['y'] - clipped['y'].iloc[0]
                ylabel = 'y displacement (m)'
            
            ax2.plot(clipped['trial_time'], position, 
                    label=f'Force={Force}', alpha=0.7)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'{"Rotation" if is_rotational else "Position"} vs Time (From Relative Zero)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Velocity-Force relationship
    if is_rotational:
        a, b, r_squared = fit_twist_force_relationship(results)
        Force = results['Force'].values
        tz = results['tz_mean'].values
        # tz_std = results['tz_std'].values
        
        # Violin plots
        violin_data = [results.iloc[i]['tz_all'] for i in range(len(results))]
        parts = ax3.violinplot(violin_data, positions=Force, widths=0.06,
                              showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('orange')
            pc.set_alpha(0.6)
        parts['bodies'][0].set_label(f"Distribution of Measured ω")
        parts['cmedians'].set_label(f"Maximum, Median, and Minimum ω")
        parts['cbars'].set_edgecolor('orange')
        parts['cbars'].set_alpha(0.6)
            
        
        # Mean points
        # ax3.errorbar(Force, tz, yerr=tz_std, fmt='o', markersize=8, 
        #             capsize=5, label='Mean ± std', alpha=0.8, color='darkred', zorder=5)
        ax3.plot(Force, tz, "o", markersize=8, label='Mean ω', color='darkred', zorder=5)
        
        # Fitted line
        Force_fit = np.linspace(Force.min() - 0.5, Force.max() + 0.5, 100)
        fit_line = a * Force_fit + b
        ax3.plot(Force_fit, fit_line, 'r-', linewidth=2, 
                label=f'Fit: ω = {a:.6f}·Torque + {b:.6f}')
        
        ax3.set_xlabel('Applied Torque (Nm)', fontsize=11)
        ax3.set_xlim([-0.05, 0.55])
        ax3.set_ylim([-40, 60])
        ax3.set_ylabel('Angular Velocity ω (deg/s)', fontsize=11)
        ax3.set_title(f'Angular Velocity-Torque Relationship (R² = {r_squared:.6f})', fontsize=12)
        
        textstr = f'a = {a:.6f} deg/(s·Nm)\nb = {b:.6f} deg/s\nR² = {r_squared:.6f}'
    else:
        a, b, r_squared = fit_velocity_force_relationship(results)
        Force = results['Force'].values
        vy = results['vy_mean'].values
        vy_std = results['vy_std'].values
        
        # Violin plots
        violin_data = [results.iloc[i]['vy_all'] for i in range(len(results))]
        parts = ax3.violinplot(violin_data, positions=Force, widths=0.3, 
                              showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightgreen')
            pc.set_alpha(0.6)
        parts['bodies'][0].set_label(f"Distribution of Measured vy")
        parts['cmedians'].set_label(f"Maximum, Median, and Minimum vy")
        parts['cbars'].set_edgecolor('lightgreen')
        parts['cbars'].set_alpha(0.6)
        
        # Mean points
        # ax3.errorbar(Force, vy, yerr=vy_std, fmt='o', markersize=8, 
        #             capsize=5, label='Mean ± std', alpha=0.8, color='darkblue', zorder=5)
        ax3.plot(Force, vy, "o", markersize=8, label='Mean vy', color='darkblue', zorder=5)


        # Fitted line
        Force_fit = np.linspace(Force.min() - 0.5, Force.max() + 0.5, 100)
        fit_line = a * Force_fit + b
        ax3.plot(Force_fit, fit_line, 'r-', linewidth=2, 
                label=f'Fit: vy = {a:.6f}·Force + {b:.6f}')
        
        ax3.set_xlabel('Applied Force (N)', fontsize=11)
        ax3.set_ylabel('Velocity vy (m/s)', fontsize=11)
        ax3.set_title(f'Velocity-Force Relationship (R² = {r_squared:.6f})', fontsize=12)
        
        textstr = f'a = {a:.6f} m/(s·N)\nb = {b:.6f} m/s\nR² = {r_squared:.6f}'
    
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add text box with fit parameters
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.15, textstr, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, results


# Example usage
if __name__ == "__main__":
    csvs = ['~/ros2_ws/src/risl_ur/risl_ur/scripts/friction_trials_smooth_jenga_6_linear.csv',
            '~/ros2_ws/src/risl_ur/risl_ur/scripts/friction_trials_smooth_jenga_6_rotational.csv']
    
    for csv in csvs:
        print(f"\n{'='*80}")
        print(f"Processing: {csv}")
        print('='*80)
        
        # Create combined figure
        fig, results = create_combined_figure(csv, t_start=1.0, t_end=5.0)
        
        # Print summary statistics
        print(f"\nSummary by Force Level:")
        print("-" * 80)
        summary = results.groupby('Force').agg({
            'vx_mean': 'mean',
            'vy_mean': 'mean',
            'tz_mean': 'mean'
        }).round(6)
        print(summary)
        
        plt.show()