"""
Stage 5 Model-Free Reinforcement Learning Foundation (Gate S5.1 & S5.2).

Exports the authoritative Gymnasium training adapter, observation encoder,
MapSuite V1 TRAIN split case sampler, and SAC trainer / checkpoint helpers.
"""

# Windows DLL initialization order: import torch before Panda3D
try:
    import torch
except ImportError:
    pass

from src.training.stage5.env_adapter import Stage5TrainingEnv
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
from src.training.stage5.checkpoint import (
    load_sac_checkpoint,
    save_sac_checkpoint,
    verify_checkpoint_integrity,
)
from src.training.stage5.sac_trainer import (
    SacBaselineTrainer,
    SacDiagnosticsCallback,
)

__all__ = [
    "Stage5TrainingEnv",
    "Stage5ObservationEncoder",
    "encode_agent_input",
    "TrainCaseSampler",
    "SamplerStrategy",
    "SacBaselineTrainer",
    "SacDiagnosticsCallback",
    "save_sac_checkpoint",
    "load_sac_checkpoint",
    "verify_checkpoint_integrity",
    "STAGE5_FULL_OBSERVATION_DIM",
    "STAGE5_CORE_ONLY_OBSERVATION_DIM",
    "STAGE5_ENCODER_ID",
    "STAGE5_ENCODER_VERSION",
    "CORE_OBSERVATION_DIM",
]
