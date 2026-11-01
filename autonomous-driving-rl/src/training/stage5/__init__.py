"""
Stage 5 Model-Free Reinforcement Learning Foundation (Gate S5.1 & S5.2).

Exports the authoritative Gymnasium training adapter, observation encoder,
and MapSuite V1 TRAIN split case sampler.
Heavy simulator and RL dependencies (Stage5TrainingEnv, SAC trainer, checkpoint utilities)
are lazily loaded on demand via PEP 562 module __getattr__ to ensure clean isolation
across lightweight evaluation, vanilla S5.1 environments, and S5.2 RL training setups.
"""

# On Windows, ensure torch DLLs initialize before launcher/Panda3D if torch is present
try:
    import torch
except (ImportError, OSError):
    pass

from src.training.stage5.observation_encoder import (
    CORE_OBSERVATION_DIM,
    STAGE5_CORE_ONLY_OBSERVATION_DIM,
    STAGE5_ENCODER_ID,
    STAGE5_ENCODER_VERSION,
    STAGE5_FULL_OBSERVATION_DIM,
    Stage5ObservationEncoder,
    encode_agent_input,
)
from src.training.stage5.train_sampler import (
    SamplerStrategy,
    TrainCaseSampler,
)

_LAZY_MODULE_EXPORTS = {
    "Stage5TrainingEnv": "src.training.stage5.env_adapter",
    "CheckpointIntegrityError": "src.training.stage5.checkpoint",
    "load_sac_checkpoint": "src.training.stage5.checkpoint",
    "save_sac_checkpoint": "src.training.stage5.checkpoint",
    "verify_checkpoint_integrity": "src.training.stage5.checkpoint",
    "SacBaselineTrainer": "src.training.stage5.sac_trainer",
    "SacDiagnosticsCallback": "src.training.stage5.sac_trainer",
}


def __getattr__(name: str):
    """Lazily loads heavy simulator adapters and RL components on demand."""
    if name in _LAZY_MODULE_EXPORTS:
        import importlib
        module = importlib.import_module(_LAZY_MODULE_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Stage5TrainingEnv",
    "Stage5ObservationEncoder",
    "encode_agent_input",
    "TrainCaseSampler",
    "SamplerStrategy",
    "STAGE5_FULL_OBSERVATION_DIM",
    "STAGE5_CORE_ONLY_OBSERVATION_DIM",
    "STAGE5_ENCODER_ID",
    "STAGE5_ENCODER_VERSION",
    "CORE_OBSERVATION_DIM",
    "SacBaselineTrainer",
    "SacDiagnosticsCallback",
    "save_sac_checkpoint",
    "load_sac_checkpoint",
    "verify_checkpoint_integrity",
    "CheckpointIntegrityError",
]
