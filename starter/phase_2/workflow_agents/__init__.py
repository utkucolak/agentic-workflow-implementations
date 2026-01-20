"""Proxy package for workflow_agents that delegates submodule lookups to phase_1.

This avoids duplication by making `workflow_agents` in phase_2 point to
the implementation located under phase_1/workflow_agents.
"""

from pathlib import Path

# Point package search path to the phase_1/workflow_agents directory
# __file__ => .../starter/phase_2/workflow_agents/__init__.py
# parents[2] => .../starter
_phase_1_workflow_agents = Path(__file__).resolve().parents[2] / "phase_1" / "workflow_agents"
__path__ = [str(_phase_1_workflow_agents)]
