"""
Stage 5 Model-Free Reinforcement Learning Foundation (Gate S5.1).

Exports the authoritative Gymnasium training adapter, observation encoder,
and MapSuite V1 TRAIN split case sampler.
"""

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
]
