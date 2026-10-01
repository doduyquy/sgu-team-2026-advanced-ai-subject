"""
Agent Registry and Registration Specifications for Research Platform V1 Launcher (Gate 7.5A).

This module implements AgentRegistryV1:
- Machine-readable AgentRegistrationV1 metadata segregated from runtime factory callables.
- Registration of Gate-6 audit/development fixture agents.
- Enforces that fixture agents are NOT benchmark-eligible (cannot run on TEST or VALIDATION).
- Provides algorithm-neutral registration interface for future Stage 0 through Stage 7 agents.
"""

from typing import Any, Callable, Dict, List, Optional

from src.launcher.models import AgentRegistrationV1
from src.platform import (
    AgentPolicy,
    DeterministicConstantFixtureAgent,
    DiscreteFixtureAgent,
    SeededRandomFixtureAgent,
    StatefulCounterFixtureAgent,
)


class AgentRegistryV1:
    """
    Authoritative registry of agents available for execution through the Research Launcher.
    Maps agent_id to machine-readable registration metadata and runtime factory callables.
    """
    def __init__(self):
        self._registrations: Dict[str, AgentRegistrationV1] = {}
        self._factories: Dict[str, Callable[[], AgentPolicy]] = {}

    def register(
        self,
        registration: AgentRegistrationV1,
        factory: Callable[[], AgentPolicy]
    ) -> None:
        """Registers an agent with its semantic capability descriptor and runtime factory."""
        if not registration.agent_id:
            raise ValueError("Cannot register an agent with empty agent_id")
        if registration.agent_id in self._registrations:
            raise ValueError(f"Agent '{registration.agent_id}' is already registered in AgentRegistryV1")

        self._registrations[registration.agent_id] = registration
        self._factories[registration.agent_id] = factory

    def get(self, agent_id: str) -> AgentRegistrationV1:
        """Retrieves semantic registration metadata for an agent."""
        if agent_id not in self._registrations:
            raise KeyError(
                f"Agent '{agent_id}' is not registered. Available agents: {list(self._registrations.keys())}"
            )
        return self._registrations[agent_id]

    def get_factory(self, agent_id: str) -> Callable[[], AgentPolicy]:
        """Retrieves runtime factory callable for an agent."""
        if agent_id not in self._factories:
            raise KeyError(
                f"Agent factory for '{agent_id}' is not found. Available agents: {list(self._factories.keys())}"
            )
        return self._factories[agent_id]

    def has_agent(self, agent_id: str) -> bool:
        """Returns True if the agent is registered."""
        return agent_id in self._registrations

    def list_all(self) -> List[AgentRegistrationV1]:
        """Returns all registered agent descriptors sorted by agent_id."""
        return [self._registrations[aid] for aid in sorted(self._registrations.keys())]

    def to_snapshot(self) -> List[Dict[str, Any]]:
        """Returns JSON-serializable list of all registered agent descriptors."""
        return [reg.to_dict() for reg in self.list_all()]


def build_default_agent_registry() -> AgentRegistryV1:
    """
    Initializes and populates AgentRegistryV1 with Gate-6 audit/development fixture agents.
    NOTE: All fixture agents are explicitly flagged with benchmark_eligible = False.
    Stage 0 Random benchmark agent will be registered in a subsequent task.
    """
    registry = AgentRegistryV1()

    # 1. Deterministic Constant Continuous Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_constant_continuous",
            agent_version="1.0.0",
            stage_label=None,
            method_family="FIXTURE",
            purpose="AUDIT_FIXTURE",
            implementation_ref="DeterministicConstantFixtureAgent",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            benchmark_eligible=False,  # Fixtures are NOT benchmark agents
            sandbox_eligible=True,
            audit_eligible=True,
            requires_checkpoint=False,
            description="Audit fixture emitting constant steering=0.0 and throttle=0.3."
        ),
        factory=lambda: DeterministicConstantFixtureAgent()
    )

    # 2. Seeded Random Continuous Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_seeded_random",
            agent_version="1.0.0",
            stage_label=None,
            method_family="FIXTURE",
            purpose="AUDIT_FIXTURE",
            implementation_ref="SeededRandomFixtureAgent",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="stochastic",
            stateful_within_episode=False,
            benchmark_eligible=False,  # NOT the final Stage-0 Random baseline
            sandbox_eligible=True,
            audit_eligible=True,
            requires_checkpoint=False,
            description="Audit fixture emitting uniform stochastic continuous actions seeded strictly via agent_seed."
        ),
        factory=lambda: SeededRandomFixtureAgent()
    )

    # 3. Stateful Counter Continuous Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_stateful_counter",
            agent_version="1.0.0",
            stage_label=None,
            method_family="FIXTURE",
            purpose="AUDIT_FIXTURE",
            implementation_ref="StatefulCounterFixtureAgent",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=True,
            benchmark_eligible=False,
            sandbox_eligible=True,
            audit_eligible=True,
            requires_checkpoint=False,
            description="Audit fixture maintaining internal step counter to verify episodic state reset."
        ),
        factory=lambda: StatefulCounterFixtureAgent()
    )

    # 4. Discrete Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_discrete",
            agent_version="1.0.0",
            stage_label=None,
            method_family="FIXTURE",
            purpose="AUDIT_FIXTURE",
            implementation_ref="DiscreteFixtureAgent",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="discrete25_native_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            benchmark_eligible=False,
            sandbox_eligible=True,
            audit_eligible=True,
            requires_checkpoint=False,
            description="Audit fixture emitting constant discrete action index 12 (stay still / idle)."
        ),
        factory=lambda: DiscreteFixtureAgent()
    )

    return registry
