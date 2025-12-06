"""
Central place to configure where OSF data lives (DNN RDMs, behavior, brain RDMs).

Usage
-----
Either set the environment variables below before running the scripts
or edit the DEFAULT_* values. All DNN scripts import helpers here so you
only need to change paths once.
"""

import os

# Default roots for OSF downloads on this machine.
DEFAULT_OSF_ROOT = "/home/clemens-uva/fs4/fMRI_OSF_data"
DEFAULT_DNN_RDM_ROOT = os.path.join(DEFAULT_OSF_ROOT, "DNN_RDMS")


def get_osf_root() -> str:
    """Base directory where all OSF data was downloaded."""
    return os.environ.get("OSF_ROOT", DEFAULT_OSF_ROOT)


def get_dnn_rdm_root() -> str:
    """Base directory containing the DNN RDM folders."""
    return os.environ.get("DNN_RDM_ROOT", DEFAULT_DNN_RDM_ROOT)


def dnn_rdm_path(*parts: str) -> str:
    """Build a path under the DNN RDM root (e.g., 'AlexNet_VISACT_RDM')."""
    return os.path.join(get_dnn_rdm_root(), *parts)


def behavior_path(*parts: str) -> str:
    """Paths under VISACT_behavior (e.g., 'VISACT_fmri_behavior', 'file.npz')."""
    return os.path.join(get_osf_root(), "VISACT_behavior", *parts)


def brain_path(*parts: str) -> str:
    """Paths under VISACT_brain_data (e.g., 'average', 'fmri_PPA_mean.npz')."""
    return os.path.join(get_osf_root(), "VISACT_brain_data", *parts)


def rdm_collection_path(*parts: str) -> str:
    """Paths under VISACT_RDM_collection (online behavior, GIST, CHAT_GPT, etc.)."""
    return os.path.join(get_osf_root(), "VISACT_RDM_collection", *parts)


def embedding_rdms_path(*parts: str) -> str:
    """Paths under Embedding_space_rdms (e.g., MiniLM embeddings)."""
    return os.path.join(get_osf_root(), "Embedding_space_rdms", *parts)
