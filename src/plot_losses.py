"""
Script to extract loss values from TensorFlow events file and create a combined plot.
The combined plot will show:
    - both training and validation losses for easy comparison
    - x-axis normalized to epoch scale (TB logs uses iteration steps)
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
    
    # Extract train loss.
    train_loss_tags = [tag for tag in scalar_tags if 'train' in tag.lower() and 'loss' in tag.lower()]
    print(f"Train loss tags: {train_loss_tags}")
    
    # Extract validation loss.
    val_loss_tags = [tag for tag in scalar_tags if 'val' in tag.lower() and 'loss' in tag.lower()]
    print(f"Validation loss tags: {val_loss_tags}")
    
    # Get the actual loss values.
    if train_loss_tags:
        events = ea.Scalars(train_loss_tags[0])
        train_losses = [(event.step, event.value) for event in events]
    
    if val_loss_tags:
        events = ea.Scalars(val_loss_tags[0])
        val_losses = [(event.step, event.value) for event in events]
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'max_epoch': max_epoch
    }


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
        steps, values = zip(*train_losses)
        # Normalize steps to epoch scale.
        epochs = np.array(steps) / (max(steps) / max_epoch) if max(steps) > 0 else np.array(steps)
        ax.plot(epochs, values, label='Train Loss', linewidth=2, marker='o', markersize=3, alpha=0.8)
    
    # Plot validation losses.
    if val_losses:
        steps, values = zip(*val_losses)
        # Normalize steps to epoch scale.
        epochs = np.array(steps) / (max(steps) / max_epoch) if max(steps) > 0 else np.array(steps)
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


if __name__ == '__main__':

    # Path to events file.
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
    
    # Create plot.
    output_path = output_dir / 'version_10_merged_loss_plot.png'
    create_combined_loss_plot(data, output_path)
    
    print("Done!")
