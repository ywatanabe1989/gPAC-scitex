def plot_frequency_band_changes(model, output_dir):
    """
    Plot changes in frequency bands from initial to current state.
    Only applicable for trainable PAC modules.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get frequency changes
    changes = model.get_frequency_changes()
    if changes is None:
        print("Not a trainable PAC module, skipping frequency band change visualization")
        return
    
    # Unpack data
    initial_pha = changes['initial_pha'].numpy()
    initial_amp = changes['initial_amp'].numpy()
    current_pha = changes['current_pha'].numpy()
    current_amp = changes['current_amp'].numpy()
    pha_changes = changes['pha_changes'].numpy()
    amp_changes = changes['amp_changes'].numpy()
    
    # Plot phase frequency changes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    
    # X-axis for indices
    x = np.arange(len(initial_pha))
    
    # Plot initial and current values
    plt.plot(x, initial_pha, 'o-', label='Initial', alpha=0.7)
    plt.plot(x, current_pha, 's-', label='Trained', alpha=0.7)
    
    # Highlight changes with arrows
    for i in range(len(initial_pha)):
        if abs(pha_changes[i]) > 0.1:  # Only show significant changes
            plt.arrow(i, initial_pha[i], 0, pha_changes[i], 
                     length_includes_head=True, head_width=0.15, 
                     head_length=abs(pha_changes[i])*0.2, 
                     fc='red', ec='red', alpha=0.6)
    
    plt.title('Phase Frequency Band Changes')
    plt.xlabel('Band Index')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot amplitude frequency changes
    plt.subplot(1, 2, 2)
    
    # X-axis for indices
    x = np.arange(len(initial_amp))
    
    # Plot initial and current values
    plt.plot(x, initial_amp, 'o-', label='Initial', alpha=0.7)
    plt.plot(x, current_amp, 's-', label='Trained', alpha=0.7)
    
    # Highlight changes with arrows
    for i in range(len(initial_amp)):
        if abs(amp_changes[i]) > 0.5:  # Only show significant changes
            plt.arrow(i, initial_amp[i], 0, amp_changes[i], 
                     length_includes_head=True, head_width=0.15, 
                     head_length=abs(amp_changes[i])*0.1, 
                     fc='red', ec='red', alpha=0.6)
    
    plt.title('Amplitude Frequency Band Changes')
    plt.xlabel('Band Index')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_band_changes.png", dpi=300)
    plt.close()
    
    # Plot frequency bands in 2D space
    plt.figure(figsize=(10, 8))
    
    # Get feature importance for marker sizes
    importance_map = model.get_feature_importance().cpu().numpy()
    importance_flat = importance_map.flatten()
    max_importance = importance_flat.max()
    
    # Create a 2D scatter plot of frequency bands
    for p_idx, pha_freq in enumerate(current_pha):
        for a_idx, amp_freq in enumerate(current_amp):
            importance = importance_map[p_idx, a_idx]
            # Size proportional to importance
            size = (importance / max_importance) * 300 + 20
            # Color based on importance
            color = plt.cm.viridis(importance / max_importance)
            
            # Plot current frequency band
            plt.scatter(pha_freq, amp_freq, s=size, color=color, alpha=0.7, 
                       edgecolor='k', linewidth=1)
            
            # Plot arrow from initial to current if there's significant change
            if (abs(current_pha[p_idx] - initial_pha[p_idx]) > 0.1 or 
                abs(current_amp[a_idx] - initial_amp[a_idx]) > 0.5):
                plt.arrow(initial_pha[p_idx], initial_amp[a_idx], 
                         current_pha[p_idx] - initial_pha[p_idx],
                         current_amp[a_idx] - initial_amp[a_idx],
                         width=0.1, head_width=0.5, head_length=1.0,
                         fc='red', ec='red', alpha=0.4)
    
    # Add class region indicators (based on generation parameters)
    # Class A region (lower-left)
    plt.axvspan(4, 8, ymin=0, ymax=0.4, alpha=0.1, color='blue', label='Class A Region')
    plt.axhspan(80, 100, xmin=0.2, xmax=0.4, alpha=0.1, color='blue')
    
    # Class B region (upper-right)
    plt.axvspan(10, 14, ymin=0.6, ymax=1.0, alpha=0.1, color='red', label='Class B Region')
    plt.axhspan(120, 150, xmin=0.5, xmax=0.7, alpha=0.1, color='red')
    
    plt.title('2D Frequency Band Map with Importances')
    plt.xlabel('Phase Frequency (Hz)')
    plt.ylabel('Amplitude Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_band_map.png", dpi=300)
    plt.close()
    
    # Save data for further analysis
    np.savez(
        output_dir / "frequency_bands_data.npz",
        initial_pha=initial_pha,
        initial_amp=initial_amp,
        current_pha=current_pha,
        current_amp=current_amp,
        pha_changes=pha_changes,
        amp_changes=amp_changes,
        importance_map=importance_map
    )