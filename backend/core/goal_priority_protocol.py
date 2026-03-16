from __future__ import annotations

from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class ConflictResolutionResult:
    winner: str
    loser: str
    reason: str
    priority_winner: int
    priority_loser: int


@dataclass(frozen=True)
class IterationAbortResult:
    should_abort: bool
    reason: str
    degraded_goals: list[str] = field(default_factory=list)


class GoalPriorityProtocol:
    PRIORITY_MAP: dict[str, int] = {
        "natuerlichkeit": 1,
        "authentizitaet": 1,
        "tonal_center": 2,
        "timbre_authentizitaet": 2,
        "artikulation": 2,
        "emotionalitaet": 3,
        "micro_dynamics": 3,
        "groove": 3,
        "transparenz": 4,
        "waerme": 4,
        "bass_kraft": 4,
        "separation_fidelity": 4,
        "brillanz": 5,
        "spatial_depth": 5,
    }

    ABORT_PRIORITY_THRESHOLD: int = 2
    REGRESSION_EPSILON: float = 0.001

    def resolve_conflict(
        self,
        goal_a: str,
        goal_b: str,
        delta_a: float,
        delta_b: float,
        headroom_a: float = 0.0,
        headroom_b: float = 0.0,
    ) -> ConflictResolutionResult:
        prio_a = self.priority_of(goal_a)
        prio_b = self.priority_of(goal_b)

        if prio_a < prio_b:
            return ConflictResolutionResult(goal_a, goal_b, "higher-priority goal wins", prio_a, prio_b)
        if prio_b < prio_a:
            return ConflictResolutionResult(goal_b, goal_a, "higher-priority goal wins", prio_b, prio_a)

        if headroom_a > headroom_b:
            return ConflictResolutionResult(goal_a, goal_b, "equal priority, higher headroom", prio_a, prio_b)
        if headroom_b > headroom_a:
            return ConflictResolutionResult(goal_b, goal_a, "equal priority, higher headroom", prio_b, prio_a)

        if delta_a >= delta_b:
            return ConflictResolutionResult(goal_a, goal_b, "equal priority/headroom, larger delta", prio_a, prio_b)
        return ConflictResolutionResult(goal_b, goal_a, "equal priority/headroom, larger delta", prio_b, prio_a)

    def should_abort_iteration(
        self,
        scores_before: dict[str, float],
        scores_after: dict[str, float],
    ) -> IterationAbortResult:
        degraded: list[str] = []
        for goal, before in scores_before.items():
            after = scores_after.get(goal, before)
            if before - after > self.REGRESSION_EPSILON and self.priority_of(goal) <= self.ABORT_PRIORITY_THRESHOLD:
                degraded.append(goal)

        if degraded:
            return IterationAbortResult(True, "critical goal regression", degraded)
        return IterationAbortResult(False, "ok", [])

    def priority_of(self, goal: str) -> int:
        return int(self.PRIORITY_MAP.get(goal, 5))

    def sort_goals_by_priority(self, goals: list[str]) -> list[str]:
        return sorted(goals, key=self.priority_of)

    def goals_at_priority(self, level: int) -> list[str]:
        return [g for g, p in self.PRIORITY_MAP.items() if p == level]

    def would_violate_priority(self, improving_goal: str, at_cost_of: str) -> bool:
        return self.priority_of(improving_goal) > self.priority_of(at_cost_of)

    def user_message_for_failure(self, goal: str) -> str:
        if self.priority_of(goal) <= 2:
            return (
                "Die Restaurierung konnte zentrale Klangziele nicht voll erreichen. "
                "Das bestmoegliche Ergebnis wurde dennoch ausgegeben."
            )
        return (
            "Einige zusaetzliche Klangziele konnten nicht voll erreicht werden. "
            "Das ist bei diesem Material physikalisch bedingt."
        )


_instance: GoalPriorityProtocol | None = None
_lock = threading.Lock()


def get_goal_priority_protocol() -> GoalPriorityProtocol:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GoalPriorityProtocol()
    return _instance


def resolve_goal_conflict(
    goal_a: str,
    goal_b: str,
    delta_a: float,
    delta_b: float,
    headroom_a: float = 0.0,
    headroom_b: float = 0.0,
) -> ConflictResolutionResult:
    return get_goal_priority_protocol().resolve_conflict(goal_a, goal_b, delta_a, delta_b, headroom_a, headroom_b)


def check_iteration_abort(
    scores_before: dict[str, float],
    scores_after: dict[str, float],
) -> IterationAbortResult:
    return get_goal_priority_protocol().should_abort_iteration(scores_before, scores_after)


__all__ = [
    "GoalPriorityProtocol",
    "ConflictResolutionResult",
    "IterationAbortResult",
    "get_goal_priority_protocol",
    "resolve_goal_conflict",
    "check_iteration_abort",
]
