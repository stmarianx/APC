from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .models import Card
from .solutions import ActionSolution, SolutionKey, SolutionStore, SolvedSpot


SUPPORTED_SCHEMA_VERSION = "1.0.0"


class SolverBundleError(ValueError):
    pass


@dataclass(frozen=True)
class SolverBundle:
    schema_version: str
    source: str
    source_version: str
    spots: tuple[SolvedSpot, ...]
    node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise SolverBundleError(f"Unsupported solver bundle schema: {self.schema_version}")
        if not self.source or not self.source_version:
            raise SolverBundleError("Solver bundle requires source and source_version")
        if not self.spots or len(self.spots) != len(self.node_ids):
            raise SolverBundleError("Solver bundle requires matched spots and node ids")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise SolverBundleError("Solver node ids must be unique")
        fingerprints = [spot.key.fingerprint for spot in self.spots]
        if len(set(fingerprints)) != len(fingerprints):
            raise SolverBundleError("Solver bundle contains duplicate solution fingerprints")


@dataclass(frozen=True)
class SolverImportResult:
    source: str
    source_version: str
    inserted: int
    updated: int
    fingerprints: tuple[str, ...]
    node_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_version": self.source_version,
            "inserted": self.inserted,
            "updated": self.updated,
            "spots": len(self.fingerprints),
            "fingerprints": list(self.fingerprints),
            "node_ids": list(self.node_ids),
        }


def solved_spot_to_dict(spot: SolvedSpot) -> dict[str, object]:
    return {
        "node_id": spot.node_id,
        "fingerprint": spot.key.fingerprint,
        "key": spot.key.canonical(),
        "actions": [
            {
                "action": action.action,
                "frequency": format(action.frequency, "f"),
                "ev": format(action.ev, "f"),
                "ev_loss": format(spot.ev_loss(action.action), "f"),
            }
            for action in spot.actions
        ],
        "best_ev": format(spot.best_ev, "f"),
        "source": spot.source,
        "source_version": spot.source_version,
    }


class SolverBundleImporter:
    def parse_file(self, path: str | Path) -> SolverBundle:
        return self.parse_text(Path(path).read_text(encoding="utf-8-sig"))

    def parse_text(self, source: str) -> SolverBundle:
        try:
            payload = json.loads(source)
        except json.JSONDecodeError as error:
            raise SolverBundleError(f"Invalid solver bundle JSON at line {error.lineno}: {error.msg}") from error
        if not isinstance(payload, dict):
            raise SolverBundleError("Solver bundle root must be an object")
        return self.parse_dict(payload)

    def parse_dict(self, payload: dict[str, Any]) -> SolverBundle:
        schema = self._required_text(payload, "schema_version", "bundle")
        if schema != SUPPORTED_SCHEMA_VERSION:
            raise SolverBundleError(f"Unsupported solver bundle schema: {schema}")
        source = self._required_text(payload, "source", "bundle")
        source_version = self._required_text(payload, "source_version", "bundle")
        rows = payload.get("spots")
        if not isinstance(rows, list) or not rows:
            raise SolverBundleError("Solver bundle spots must be a non-empty array")
        spots: list[SolvedSpot] = []
        node_ids: list[str] = []
        for index, row in enumerate(rows):
            context = f"spots[{index}]"
            if not isinstance(row, dict):
                raise SolverBundleError(f"{context} must be an object")
            node_id = self._required_text(row, "node_id", context)
            key_data = row.get("key")
            if not isinstance(key_data, dict):
                raise SolverBundleError(f"{context}.key must be an object")
            key = self._parse_key(key_data, f"{context}.key")
            actions_data = row.get("actions")
            if not isinstance(actions_data, list) or not actions_data:
                raise SolverBundleError(f"{context}.actions must be a non-empty array")
            actions: list[ActionSolution] = []
            for action_index, action_data in enumerate(actions_data):
                action_context = f"{context}.actions[{action_index}]"
                if not isinstance(action_data, dict):
                    raise SolverBundleError(f"{action_context} must be an object")
                try:
                    actions.append(
                        ActionSolution(
                            action=self._required_text(action_data, "action", action_context),
                            frequency=self._decimal(action_data.get("frequency"), f"{action_context}.frequency"),
                            ev=self._decimal(action_data.get("ev"), f"{action_context}.ev"),
                        )
                    )
                except ValueError as error:
                    raise SolverBundleError(f"{action_context}: {error}") from error
            try:
                spot = SolvedSpot(
                    key=key,
                    actions=tuple(actions),
                    source=source,
                    source_version=source_version,
                    node_id=node_id,
                )
            except ValueError as error:
                raise SolverBundleError(f"{context}: {error}") from error
            node_ids.append(node_id)
            spots.append(spot)
        return SolverBundle(schema, source, source_version, tuple(spots), tuple(node_ids))

    def import_into(self, store: SolutionStore, bundle: SolverBundle) -> SolverImportResult:
        inserted = updated = 0
        for spot in bundle.spots:
            if store.get(spot.key) is None:
                inserted += 1
            else:
                updated += 1
            store.put(spot)
        return SolverImportResult(
            source=bundle.source,
            source_version=bundle.source_version,
            inserted=inserted,
            updated=updated,
            fingerprints=tuple(spot.key.fingerprint for spot in bundle.spots),
            node_ids=bundle.node_ids,
        )

    def _parse_key(self, data: dict[str, Any], context: str) -> SolutionKey:
        try:
            players = int(data.get("players"))
        except (TypeError, ValueError) as error:
            raise SolverBundleError(f"{context}.players must be an integer") from error
        board = self._cards(data.get("board", []), f"{context}.board")
        hero_cards = self._cards(data.get("hero_cards", []), f"{context}.hero_cards")
        action_history = data.get("action_history", [])
        allowed_sizes = data.get("allowed_sizes", [])
        if not isinstance(action_history, list) or not all(isinstance(item, str) for item in action_history):
            raise SolverBundleError(f"{context}.action_history must be a string array")
        if not isinstance(allowed_sizes, list):
            raise SolverBundleError(f"{context}.allowed_sizes must be an array")
        try:
            return SolutionKey(
                game=self._required_text(data, "game", context),
                players=players,
                hero_position=self._required_text(data, "hero_position", context),
                effective_stack_bb=self._decimal(data.get("effective_stack_bb"), f"{context}.effective_stack_bb"),
                pot_bb=self._decimal(data.get("pot_bb"), f"{context}.pot_bb"),
                board=board,
                action_history=tuple(action_history),
                rake_model=self._required_text(data, "rake_model", context),
                utility_model=str(data.get("utility_model", "chip_ev")),
                allowed_sizes=tuple(
                    self._decimal(size, f"{context}.allowed_sizes[{index}]")
                    for index, size in enumerate(allowed_sizes)
                ),
                hero_cards=hero_cards,
            )
        except ValueError as error:
            if isinstance(error, SolverBundleError):
                raise
            raise SolverBundleError(f"{context}: {error}") from error

    @staticmethod
    def _required_text(data: dict[str, Any], key: str, context: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SolverBundleError(f"{context}.{key} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _decimal(value: object, context: str) -> Decimal:
        if isinstance(value, bool) or value is None:
            raise SolverBundleError(f"{context} must be a decimal number")
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise SolverBundleError(f"{context} must be a decimal number") from error
        if not result.is_finite():
            raise SolverBundleError(f"{context} must be finite")
        return result

    @staticmethod
    def _cards(value: object, context: str) -> tuple[Card, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise SolverBundleError(f"{context} must be a card-string array")
        try:
            return tuple(Card.parse(item) for item in value)
        except ValueError as error:
            raise SolverBundleError(f"{context}: {error}") from error
