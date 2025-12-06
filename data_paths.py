"""
Repository-wide data path helpers.

Set the environment variable DATA_ROOT to point to your OSF download
directory; otherwise the default below is used.
"""

import os

DEFAULT_DATA_ROOT = "/home/clemens-uva/fs4/fMRI_OSF_data"


def get_data_root() -> str:
    """Base directory where OSF data is stored."""
    return os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)


def searchlight_base_path() -> str:
    """Directory containing Searchlight_spearman_base results."""
    return os.path.join(get_data_root(), "Searchlight_spearman_base")


def searchlight_partial_path() -> str:
    """Directory containing Searchlight_partial_newest results."""
    return os.path.join(get_data_root(), "Searchlight_partial_newest")


def bids_output_path() -> str:
    """BIDS derivatives root used in searchlight analyses."""
    return os.path.join(get_data_root(), "VISACT_out")


def rdm_collection_path(*parts: str) -> str:
    """Paths under VISACT_RDM_collection."""
    return os.path.join(get_data_root(), "VISACT_RDM_collection", *parts)


def behavior_path(*parts: str) -> str:
    """Paths under VISACT_behavior (e.g., VISACT_fmri_behavior files)."""
    return os.path.join(get_data_root(), "VISACT_behavior", *parts)


def brain_path(*parts: str) -> str:
    """Paths under VISACT_brain_data."""
    return os.path.join(get_data_root(), "VISACT_brain_data", *parts)


def embedding_rdms_path(*parts: str) -> str:
    """Paths under Embedding_space_rdms (e.g., MiniLM embeddings)."""
    return os.path.join(get_data_root(), "Embedding_space_rdms", *parts)


def dnn_rdm_path(*parts: str) -> str:
    """Paths under DNN_RDMS."""
    return os.path.join(get_data_root(), "DNN_RDMS", *parts)
