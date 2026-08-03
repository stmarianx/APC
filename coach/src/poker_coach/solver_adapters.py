from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from .solver_import import SolverBundle, SolverBundleError, SolverBundleImporter


BUNDLE_JSON_V1 = "bundle-json-v1"
TABULAR_CSV_V1 = "tabular-csv-v1"
SUPPORTED_EXPORT_FORMATS = (BUNDLE_JSON_V1, TABULAR_CSV_V1)

CSV_COLUMNS = (
    "schema_version",
    "source",
    "source_version",
    "node_id",
    "game",
    "players",
    "hero_position",
    "effective_stack_bb",
    "pot_bb",
    "board",
    "hero_cards",
    "action_history",
    "rake_model",
    "utility_model",
    "allowed_sizes",
    "action",
    "frequency",
    "ev",
)


@dataclass(frozen=True)
class ParsedSolverExport:
    format_name: str
    bundle: SolverBundle


def _split_cards(value: str) -> list[str]:
    return value.replace("|", " ").split()


def _split_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


class TabularSolverCSVAdapter:
    format_name = TABULAR_CSV_V1

    def __init__(self, bundle_importer: SolverBundleImporter | None = None) -> None:
        self.bundle_importer = bundle_importer or SolverBundleImporter()

    def parse_text(self, source: str) -> SolverBundle:
        if not source.strip():
            raise SolverBundleError("Tabular solver CSV cannot be empty")
        reader = csv.DictReader(io.StringIO(source.lstrip("\ufeff")))
        headers = reader.fieldnames
        if not headers:
            raise SolverBundleError("Tabular solver CSV requires a header row")
        if len(headers) != len(set(headers)):
            raise SolverBundleError("Tabular solver CSV contains duplicate column names")
        missing = [column for column in CSV_COLUMNS if column not in headers]
        if missing:
            raise SolverBundleError(
                "Tabular solver CSV is missing columns: " + ", ".join(missing)
            )

        groups: dict[str, dict[str, object]] = {}
        source_name: str | None = None
        source_version: str | None = None
        schema_version: str | None = None
        row_count = 0
        key_columns = CSV_COLUMNS[4:15]
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise SolverBundleError(
                    f"Tabular solver CSV line {line_number} has more fields than headers"
                )
            if not any(str(value or "").strip() for value in row.values()):
                continue
            row_count += 1
            cleaned = {key: str(value or "").strip() for key, value in row.items()}
            for required in (
                "schema_version",
                "source",
                "source_version",
                "node_id",
                "game",
                "players",
                "hero_position",
                "effective_stack_bb",
                "pot_bb",
                "hero_cards",
                "rake_model",
                "action",
                "frequency",
                "ev",
            ):
                if not cleaned[required]:
                    raise SolverBundleError(
                        f"Tabular solver CSV line {line_number} requires {required}"
                    )
            identity = (
                cleaned["schema_version"],
                cleaned["source"],
                cleaned["source_version"],
            )
            if schema_version is None:
                schema_version, source_name, source_version = identity
            elif identity != (schema_version, source_name, source_version):
                raise SolverBundleError(
                    f"Tabular solver CSV line {line_number} changes bundle provenance"
                )

            node_id = cleaned["node_id"]
            key_signature = tuple(cleaned[column] for column in key_columns)
            group = groups.get(node_id)
            if group is None:
                group = {
                    "signature": key_signature,
                    "line": line_number,
                    "key": {
                        "game": cleaned["game"],
                        "players": cleaned["players"],
                        "hero_position": cleaned["hero_position"],
                        "effective_stack_bb": cleaned["effective_stack_bb"],
                        "pot_bb": cleaned["pot_bb"],
                        "board": _split_cards(cleaned["board"]),
                        "hero_cards": _split_cards(cleaned["hero_cards"]),
                        "action_history": _split_pipe(cleaned["action_history"]),
                        "rake_model": cleaned["rake_model"],
                        "utility_model": cleaned["utility_model"] or "chip_ev",
                        "allowed_sizes": _split_pipe(cleaned["allowed_sizes"]),
                    },
                    "actions": [],
                }
                groups[node_id] = group
            elif group["signature"] != key_signature:
                raise SolverBundleError(
                    f"Tabular solver CSV line {line_number} changes key fields for node {node_id!r}"
                )
            actions = group["actions"]
            assert isinstance(actions, list)
            actions.append(
                {
                    "action": cleaned["action"],
                    "frequency": cleaned["frequency"],
                    "ev": cleaned["ev"],
                }
            )

        if row_count == 0 or not groups:
            raise SolverBundleError("Tabular solver CSV contains no action rows")
        assert schema_version is not None and source_name is not None and source_version is not None
        payload = {
            "schema_version": schema_version,
            "source": source_name,
            "source_version": source_version,
            "spots": [
                {
                    "node_id": node_id,
                    "key": group["key"],
                    "actions": group["actions"],
                }
                for node_id, group in groups.items()
            ],
        }
        return self.bundle_importer.parse_dict(payload)


class SolverExportRegistry:
    def __init__(self, bundle_importer: SolverBundleImporter | None = None) -> None:
        self.bundle_importer = bundle_importer or SolverBundleImporter()
        self.csv_adapter = TabularSolverCSVAdapter(self.bundle_importer)

    def detect_format(self, source: str, *, path_hint: str | Path | None = None) -> str:
        suffix = Path(path_hint).suffix.lower() if path_hint is not None else ""
        if suffix == ".json":
            return BUNDLE_JSON_V1
        if suffix == ".csv":
            return TABULAR_CSV_V1
        stripped = source.lstrip("\ufeff \t\r\n")
        if stripped.startswith("{"):
            return BUNDLE_JSON_V1
        first_line = stripped.splitlines()[0] if stripped else ""
        if "node_id" in first_line and "frequency" in first_line and "," in first_line:
            return TABULAR_CSV_V1
        raise SolverBundleError("Cannot detect solver export format")

    def parse_text(
        self,
        source: str,
        *,
        format_name: str = "auto",
        path_hint: str | Path | None = None,
    ) -> ParsedSolverExport:
        selected = (
            self.detect_format(source, path_hint=path_hint)
            if format_name == "auto"
            else format_name
        )
        if selected == BUNDLE_JSON_V1:
            bundle = self.bundle_importer.parse_text(source)
        elif selected == TABULAR_CSV_V1:
            bundle = self.csv_adapter.parse_text(source)
        else:
            raise SolverBundleError(
                f"Unsupported solver export format: {selected}; expected one of "
                + ", ".join(SUPPORTED_EXPORT_FORMATS)
            )
        return ParsedSolverExport(selected, bundle)

    def parse_file(
        self, path: str | Path, *, format_name: str = "auto"
    ) -> ParsedSolverExport:
        file_path = Path(path)
        return self.parse_text(
            file_path.read_text(encoding="utf-8-sig"),
            format_name=format_name,
            path_hint=file_path,
        )
