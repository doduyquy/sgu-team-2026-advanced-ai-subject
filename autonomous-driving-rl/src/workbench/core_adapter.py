"""
Workbench Core Adapter (Gate 7.5B).

Read-only informational bridge from Gate 7.5A core into the Workbench GUI.
Provides metadata for UI controls and explorers without modifying any scientific authority:
- Available modes from LauncherMode.
- Registered agent metadata from AgentRegistryV1.
- Case manifest inspection for Case Explorer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.launcher.cases import (
    LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256,
    LOCKED_TEST_CASE_MANIFEST_SHA256,
    LOCKED_VALIDATION_CASE_MANIFEST_SHA256,
    get_default_project_root,
    load_manifest_csv,
)
from src.launcher.models import AgentRegistrationV1, LauncherMode
from src.launcher.registry import build_canonical_agent_registry


class CoreAdapter:
    """Read-only adapter providing inspection data from Gate 7.5A to the Workbench UI."""

    def __init__(self, project_root: Optional[Path] = None):
        self._project_root = project_root or get_default_project_root()
        self._registry = build_canonical_agent_registry()

    @property
    def project_root(self) -> Path:
        return self._project_root

    def get_supported_modes(self) -> List[str]:
        return [m.value for m in LauncherMode]

    def get_registered_agents(self) -> List[AgentRegistrationV1]:
        return self._registry.list_all()

    def get_agent_registration(self, agent_id: str) -> Optional[AgentRegistrationV1]:
        try:
            return self._registry.get(agent_id)
        except KeyError:
            return None

    def load_case_manifest(self, split: str) -> List[Dict[str, Any]]:
        """
        Loads frozen manifest rows for read-only browsing in Case Explorer.
        split: 'TRAIN', 'VALIDATION', or 'TEST'.
        Uses authoritative Gate-5 manifest paths and locked cryptographic hashes.
        """
        split_norm = split.upper()
        audit_manifest_dir = self._project_root / "results" / "audits" / "evaluation_protocol"
        if split_norm == "TRAIN":
            manifest_path = audit_manifest_dir / "geometry_split_manifest.csv"
            raw_cases = load_manifest_csv(manifest_path, expected_sha256=LOCKED_GEOMETRY_SPLIT_MANIFEST_SHA256)
            return [c for c in raw_cases if c.get("split") == "TRAIN"]
        elif split_norm == "VALIDATION":
            manifest_path = audit_manifest_dir / "validation_case_manifest.csv"
            return load_manifest_csv(manifest_path, expected_sha256=LOCKED_VALIDATION_CASE_MANIFEST_SHA256)
        elif split_norm == "TEST":
            manifest_path = audit_manifest_dir / "test_case_manifest.csv"
            return load_manifest_csv(manifest_path, expected_sha256=LOCKED_TEST_CASE_MANIFEST_SHA256)
        else:
            raise ValueError(f"Unknown split: {split}")
