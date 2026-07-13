__version__ = "0.23.1"

def __getattr__(name):
    raise RuntimeError(
        "rerun-sdk stub is installed for headless ARM openpi training. "
        "Visualization and teleoperation features that require rerun are unavailable."
    )
