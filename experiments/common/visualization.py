"""Noninteractive plot output shared by training, clustering and reporting."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def figure(path, dpi=300):
    """Save and close the current matplotlib figure.

    Args:
        path (str | Path): Destination image path; parent must exist.
        dpi (int): Export resolution, default 300 dots per inch.

    Returns:
        None: Writes an image at path.

    Processing:
        Apply tight layout, export with tight bounds, and close the figure.

    Raises:
        OSError: If the destination cannot be written.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
