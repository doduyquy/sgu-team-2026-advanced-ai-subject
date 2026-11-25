"""Stage 4 PDD hybrid: the reviewed supported40 policy, packaged for the launcher.

Uses learned ExtraTrees trajectories/speeds plus the existing Stage 1 controller.
This candidate is sandbox/audit only; it is not a qualified standalone policy.
"""
import hashlib
import json
from pathlib import Path

from src.platform.agent import AgentDescriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / 'artifacts/agent4_pdd/1.0.0'
MANIFEST_SHA256 = '517ede2835adfc9a02a6ea664053512163d66dfab1f8e24bdc9f800ec26ac649'


def verify_release():
    """Check the release before importing/unpickling its trusted checkpoint."""
    from importlib.metadata import version
    manifest_path = RELEASE_ROOT/'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Agent4 release manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['schema'] != 'STAGE4_PDD_HYBRID_RELEASE_V1':
        raise ValueError('Unsupported Agent4 release manifest')
    for relative, expected in manifest['files_sha256'].items():
        path = (PROJECT_ROOT/relative).resolve()
        if PROJECT_ROOT not in path.parents:
            raise ValueError('Agent4 manifest path escapes the project')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Agent4 release file checksum mismatch: {relative}')
    if version('scikit-learn') != manifest['recorded_dependencies']['scikit-learn']:
        raise ValueError('Agent4 requires the checkpoint scikit-learn version: '
                         + manifest['recorded_dependencies']['scikit-learn'])
    return manifest


class Stage4PDDHybridAgent:
    """AgentPolicy interface, matching the Stage 0/1 entrypoint lifecycle."""
    def __init__(self):
        from src.agents.agent4_pdd_loader import load_frozen_runtime
        self.release = verify_release()
        runtime = load_frozen_runtime()
        self._policy = runtime.SupportedMovingRouteDiagnostic(
            checkpoint_report=RELEASE_ROOT/'training_report.json',
            expected_sha=self.release['training_report_sha256'])
        self.descriptor = AgentDescriptor(agent_id='stage4_pdd_hybrid', agent_version='1.0.0',
                                          stateful_within_episode=True, method_family='HYBRID')

    def reset(self, public_context, agent_seed=None):
        return self._policy.reset(public_context, agent_seed)

    def act(self, agent_input):
        return self._policy.act(agent_input)

    def close(self):
        return self._policy.close()

    def export_config(self):
        return {**self._policy.export_config(), 'release_version': '1.0.0',
            'release_manifest_sha256': MANIFEST_SHA256,
            'entrypoint_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_location': 'artifacts/agent4_pdd/1.0.0/model_bundle.joblib',
            'dataset_workspace_required': False,
            'recorded_dependencies': self.release['recorded_dependencies'],
            'historical_evidence': {'medium_hard_clean': '5/5', 'extreme_clean': '0/3'},
            'integration_runtime_verified': False}
