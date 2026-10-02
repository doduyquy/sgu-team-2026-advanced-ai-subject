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
    canonical_json_sha256,
)


class AgentRegistryV1:
    """
    Authoritative registry of agents available for execution through the Research Launcher.
    Maps agent_id to machine-readable registration metadata and runtime factory callables.
    """
    def __init__(self):
        self._registrations: Dict[str, AgentRegistrationV1] = {}
        self._factories: Dict[str, Callable[[], AgentPolicy]] = {}
        self._factory_refs: Dict[str, str] = {}

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

        factory_module = getattr(factory, "__module__", "unknown")
        factory_qualname = getattr(factory, "__qualname__", getattr(factory, "__name__", "unknown"))
        factory_ref = f"{factory_module}:{factory_qualname}"

        self._registrations[registration.agent_id] = registration
        self._factories[registration.agent_id] = factory
        self._factory_refs[registration.agent_id] = factory_ref

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

    def get_factory_ref(self, agent_id: str) -> str:
        """Retrieves machine-verifiable factory reference (module:qualname) for an agent."""
        if agent_id not in self._factory_refs:
            raise KeyError(
                f"Agent factory ref for '{agent_id}' is not found. Available agents: {list(self._factory_refs.keys())}"
            )
        return self._factory_refs[agent_id]

    def has_agent(self, agent_id: str) -> bool:
        """Returns True if the agent is registered."""
        return agent_id in self._registrations

    def list_all(self) -> List[AgentRegistrationV1]:
        """Returns all registered agent descriptors sorted by agent_id."""
        return [self._registrations[aid] for aid in sorted(self._registrations.keys())]

    def to_snapshot(self) -> List[Dict[str, Any]]:
        """Returns JSON-serializable list of all registered agent descriptors."""
        return [reg.to_dict() for reg in self.list_all()]


def _factory_fixture_constant_continuous() -> DeterministicConstantFixtureAgent:
    return DeterministicConstantFixtureAgent()


def _factory_fixture_seeded_random() -> SeededRandomFixtureAgent:
    return SeededRandomFixtureAgent()


def _factory_fixture_stateful_counter() -> StatefulCounterFixtureAgent:
    return StatefulCounterFixtureAgent()


def _factory_fixture_discrete() -> DiscreteFixtureAgent:
    return DiscreteFixtureAgent()


def build_canonical_agent_registry() -> AgentRegistryV1:
    """
    Initializes and populates canonical AgentRegistryV1 with committed Gate-6 fixture agents.
    This function represents the single authoritative source of truth for Platform V1 agents.
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
            method_family="fixture",
            purpose="AUDIT_FIXTURE",
            implementation_ref=f"{DeterministicConstantFixtureAgent.__module__}:{DeterministicConstantFixtureAgent.__qualname__}",
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="deterministic",
            stateful_within_episode=False,
            benchmark_eligible=False,  # Fixtures are NOT benchmark agents
            sandbox_eligible=True,
            audit_eligible=True,
            requires_checkpoint=False,
            description="Audit fixture emitting constant steering=0.0 and throttle=0.4."
        ),
        factory=_factory_fixture_constant_continuous
    )

    # 2. Seeded Random Continuous Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_seeded_random",
            agent_version="1.0.0",
            stage_label=None,
            method_family="fixture",
            purpose="AUDIT_FIXTURE",
            implementation_ref=f"{SeededRandomFixtureAgent.__module__}:{SeededRandomFixtureAgent.__qualname__}",
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
        factory=_factory_fixture_seeded_random
    )

    # 3. Stateful Counter Continuous Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_stateful_counter",
            agent_version="1.0.0",
            stage_label=None,
            method_family="fixture",
            purpose="AUDIT_FIXTURE",
            implementation_ref=f"{StatefulCounterFixtureAgent.__module__}:{StatefulCounterFixtureAgent.__qualname__}",
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
        factory=_factory_fixture_stateful_counter
    )

    # 4. Discrete Fixture Agent
    registry.register(
        registration=AgentRegistrationV1(
            agent_id="fixture_discrete",
            agent_version="1.0.0",
            stage_label=None,
            method_family="fixture",
            purpose="AUDIT_FIXTURE",
            implementation_ref=f"{DiscreteFixtureAgent.__module__}:{DiscreteFixtureAgent.__qualname__}",
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
        factory=_factory_fixture_discrete
    )

    return registry


build_default_agent_registry = build_canonical_agent_registry


def compute_canonical_registry_sha256(registry: AgentRegistryV1) -> str:
    """
    Computes deterministic SHA-256 fingerprint over canonical AgentRegistryV1 entries.
    Covers all 14 scientific registration fields plus factory_ref for every registered agent.
    Excludes cosmetic description.
    """
    entries = []
    for reg in registry.list_all():
        factory_ref = registry.get_factory_ref(reg.agent_id)
        entries.append({
            "agent_id": reg.agent_id,
            "agent_version": reg.agent_version,
            "stage_label": reg.stage_label,
            "method_family": reg.method_family,
            "purpose": reg.purpose,
            "implementation_ref": reg.implementation_ref,
            "input_profile_id": reg.input_profile_id,
            "action_adapter_id": reg.action_adapter_id,
            "inference_stochasticity": reg.inference_stochasticity,
            "stateful_within_episode": reg.stateful_within_episode,
            "benchmark_eligible": reg.benchmark_eligible,
            "sandbox_eligible": reg.sandbox_eligible,
            "audit_eligible": reg.audit_eligible,
            "requires_checkpoint": reg.requires_checkpoint,
            "factory_ref": factory_ref
        })
    return canonical_json_sha256({"canonical_agents": entries})
