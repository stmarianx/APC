from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from apc.tools.validate_dataset import canonical_sha256


SCHEMA_VERSION = "1.0.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Artifact is outside the declared root: {path}") from error


def _files(inputs: Iterable[str | Path]) -> list[Path]:
    rows: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            rows.append(path)
        elif path.is_dir():
            rows.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise ValueError(f"Artifact input does not exist: {path}")
    return sorted(set(rows), key=lambda row: row.as_posix().casefold())


def build_manifest(
    root: str | Path,
    inputs: Iterable[str | Path],
    *,
    producer: str,
    artifact_class: str,
    source_fingerprints: dict[str, str] | None = None,
) -> dict[str, object]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"Artifact root does not exist: {base}")
    files = _files(inputs)
    if not files:
        raise ValueError("At least one artifact file is required")
    artifacts = []
    for path in files:
        relative = _inside(base, path)
        artifacts.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    material: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "apc_external_artifact_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": producer.strip(),
        "artifact_class": artifact_class.strip(),
        "source_fingerprints": dict(sorted((source_fingerprints or {}).items())),
        "artifacts": artifacts,
    }
    if not material["producer"] or not material["artifact_class"]:
        raise ValueError("producer and artifact_class must be non-empty")
    material["manifest_sha256"] = canonical_sha256(material)
    return material


def verify_manifest(root: str | Path, manifest: dict[str, object]) -> dict[str, object]:
    base = Path(root).expanduser().resolve()
    supplied = manifest.get("manifest_sha256")
    material = dict(manifest)
    material.pop("manifest_sha256", None)
    expected_manifest = canonical_sha256(material)
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if manifest.get("kind") != "apc_external_artifact_manifest":
        errors.append("unsupported manifest kind")
    if supplied != expected_manifest:
        errors.append("manifest_sha256 does not match manifest contents")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty array")
        artifacts = []
    seen: set[str] = set()
    checked = 0
    for row in artifacts:
        if not isinstance(row, dict):
            errors.append("artifact row is not an object")
            continue
        relative = str(row.get("path", ""))
        candidate = (base / relative).resolve()
        try:
            _inside(base, candidate)
        except ValueError as error:
            errors.append(str(error))
            continue
        if relative in seen:
            errors.append(f"duplicate artifact path: {relative}")
            continue
        seen.add(relative)
        if not candidate.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        checked += 1
        if candidate.stat().st_size != row.get("size_bytes"):
            errors.append(f"size mismatch: {relative}")
        if _sha256(candidate) != row.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
    return {
        "valid": not errors,
        "checked_artifacts": checked,
        "declared_artifacts": len(artifacts),
        "manifest_sha256": expected_manifest,
        "errors": errors,
    }


def _fingerprints(values: list[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source-fingerprint must use NAME=VALUE")
        name, fingerprint = value.split("=", 1)
        if not name.strip() or not fingerprint.strip():
            raise ValueError("--source-fingerprint must use non-empty NAME=VALUE")
        rows[name.strip()] = fingerprint.strip()
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify portable APC artifact manifests.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Fingerprint files for transfer to another machine")
    create.add_argument("root", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("inputs", type=Path, nargs="+")
    create.add_argument("--producer", required=True)
    create.add_argument("--artifact-class", required=True)
    create.add_argument("--source-fingerprint", action="append", default=[])
    verify = sub.add_parser("verify", help="Verify transferred files against a manifest")
    verify.add_argument("root", type=Path)
    verify.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = build_manifest(
                args.root,
                args.inputs,
                producer=args.producer,
                artifact_class=args.artifact_class,
                source_fingerprints=_fingerprints(args.source_fingerprint),
            )
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"manifest": str(output), "artifacts": len(manifest["artifacts"]), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
            return 0
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = verify_manifest(args.root, manifest)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
