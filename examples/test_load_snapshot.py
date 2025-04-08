"""Test loading a snapshot."""

from self_improving_agents.environment.snapshot import EnvironmentSnapshot

environment_snapshot = EnvironmentSnapshot()

snapshot = environment_snapshot.load()
print(snapshot)
