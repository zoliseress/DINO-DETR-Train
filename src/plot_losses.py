"""
Utilities to extract losses from TensorBoard event files and create plots.

Includes:
    - single-run train vs val loss plot
    - multi-run validation loss comparison plot
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_losses_from_events(event_file_path, max_epoch=70):
    """
    Extract train and validation losses from TensorFlow events file.
    
    Args:
        event_file_path: Path to the events.out.tfevents file
        max_epoch: Maximum epoch number to normalize x-axis
    
    Returns:
        Dict with train_losses, val_losses, and max epoch number
    """
    ea = event_accumulator.EventAccumulator(str(event_file_path))
    ea.Reload()
    
    # Get available scalar tags.
    scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags: {scalar_tags}")
    
    train_losses = []
    val_losses = []
    
    # Use the aggregate tags rather than auxiliary loss components.
    print(f"Train loss available: {'train_loss' in scalar_tags}")
    print(f"Validation loss available: {'val_loss' in scalar_tags}")
    
    # Get the actual aggregate loss values.
    if 'train_loss' in scalar_tags:
        events = ea.Scalars('train_loss')
        train_losses = [(event.step, event.value) for event in events]
    
    if 'val_loss' in scalar_tags:
        events = ea.Scalars('val_loss')
        val_losses = [(event.step, event.value) for event in events]
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'max_epoch': max_epoch
    }


def extract_single_val_loss(event_file_path):
    """Extract only val_loss scalar pairs (step, value) from one events file."""
    ea = event_accumulator.EventAccumulator(str(event_file_path))
    ea.Reload()

    scalar_tags = ea.Tags().get('scalars', [])
    if 'val_loss' not in scalar_tags:
        return []

    return [(event.step, event.value) for event in ea.Scalars('val_loss')]


def steps_to_epochs(step_value_pairs, max_epoch):
    """Normalize TensorBoard steps to an epoch axis in [0, max_epoch]."""
    if not step_value_pairs:
        return np.array([]), np.array([])

    steps, values = zip(*step_value_pairs)
    steps = np.array(steps, dtype=float)
    values = np.array(values, dtype=float)

    max_step = steps.max()
    if max_step <= 0:
        return steps, values

    epochs = steps * (max_epoch / max_step)
    return epochs, values


def create_combined_loss_plot(data, output_path):
    """
    Create a combined plot of train and validation losses.
    
    Args:
        data: Dict with train_losses and val_losses
        output_path: Path to save the plot
    """
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    max_epoch = data['max_epoch']
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    # Plot train losses.
    if train_losses:
        epochs, values = steps_to_epochs(train_losses, max_epoch)
        ax.plot(epochs, values, label='Train Loss', linewidth=2, marker='o', markersize=3, alpha=0.8)
    
    # Plot validation losses.
    if val_losses:
        epochs, values = steps_to_epochs(val_losses, max_epoch)
        ax.plot(epochs, values, label='Validation Loss', linewidth=2, marker='s', markersize=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show epoch range.
    ax.set_xlim(0, max_epoch)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def create_val_loss_comparison_plot(run_to_event_file, run_to_label, output_path, max_epoch=70):
    """
    Create one plot with validation loss curves from multiple runs.

    Args:
        run_to_event_file: mapping of run name to event file path
        run_to_label: mapping of run name to legend label
        output_path: path to save figure
        max_epoch: epoch max for x-axis
    """
    _, ax = plt.subplots(figsize=(12, 6))

    for run_name, event_file in run_to_event_file.items():
        val_loss_pairs = extract_single_val_loss(event_file)
        epochs, values = steps_to_epochs(val_loss_pairs, max_epoch)

        if len(epochs) == 0:
            print(f"Warning: no val_loss data found for {run_name}")
            continue

        ax.plot(epochs, values, linewidth=2.2, label=run_to_label.get(run_name, run_name), alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_epoch)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':

    # Path to events file for single-run train/val plot.
    events_dir = Path('lightning_logs/version_10_merged')
    event_files = list(events_dir.glob('events.out.tfevents*'))
    
    if not event_files:
        print(f"No events file found in {events_dir}")
        exit(1)
    
    event_file = event_files[0]
    print(f"Processing events file: {event_file}")
    
    # Extract losses.
    data = extract_losses_from_events(event_file, max_epoch=70)
    
    # Create output directory.
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create single-run train/val plot.
    output_path = output_dir / 'version_10_merged_loss_plot.png'
    create_combined_loss_plot(data, output_path)

    # Create multi-run validation-loss comparison plot.
    comparison_runs = {
        'version_4_merged': Path('lightning_logs/version_4_merged/events.out.tfevents.1779913677.FJ-ZSERESS-N.57832.0'),
        'version_7_merged': Path('lightning_logs/version_7_merged/events.out.tfevents.1779913261.FJ-ZSERESS-N.63324.0'),
        'version_10_merged': Path('lightning_logs/version_10_merged/events.out.tfevents.1780380751.FJ-ZSERESS-N.17152.0'),
    }
    comparison_labels = {
        'version_4_merged': 'Original DETR + ResNet-50',
        'version_7_merged': 'Conditional DETR + ResNet-50',
        'version_10_merged': 'Conditional DETR + DINOv2',
    }
    comparison_output = output_dir / 'val_loss_model_comparison.png'
    create_val_loss_comparison_plot(comparison_runs, comparison_labels, comparison_output, max_epoch=70)
    
    print("Done!")
