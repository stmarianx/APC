from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .project import AnnotationProject


ASSETS = Path(__file__).resolve().parent / "web_assets"
ANNOTATION_RE = re.compile(r"^/api/annotations/([A-Za-z0-9._-]{1,128})$")
FRAME_RE = re.compile(r"^/api/frames/([A-Za-z0-9._-]{1,128})$")
MAX_BODY_BYTES = 5 * 1024 * 1024


class AnnotatorApplication:
    def __init__(self, project: AnnotationProject) -> None:
        self.project = project
        self._lock = threading.RLock()

    def project_state(self) -> dict[str, object]:
        with self._lock:
            records = []
            for record in self.project.records:
                annotation = self.project.load_annotation(record.sample_id)
                suggestion = self.project.load_suggestion(record.sample_id)
                records.append(
                    {
                        **record.to_dict(),
                        "annotated": annotation is not None,
                        "suggested": suggestion is not None,
                        "verified": bool(
                            annotation
                            and annotation.get("provenance", {}).get("verified")
                        ),
                    }
                )
            return {
                "status": self.project.status(),
                "environment": dict(self.project.config["environment"]),
                "records": records,
            }

    def import_frames(self, payload: dict[str, object]) -> dict[str, object]:
        paths = payload.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("paths must be a non-empty array")
        session = str(payload.get("capture_session_id", "")).strip()
        timestamp_ms = int(payload.get("timestamp_ms", 0))
        rows = []
        with self._lock:
            for offset, path in enumerate(paths):
                record, inserted = self.project.import_frame(
                    str(path),
                    capture_session_id=session,
                    timestamp_ms=timestamp_ms + offset,
                )
                rows.append({**record.to_dict(), "inserted": inserted})
        return {"frames": rows, "project": self.project_state()}

    def annotation(self, sample_id: str) -> dict[str, object]:
        with self._lock:
            saved = self.project.load_annotation(sample_id)
            suggestion = self.project.load_suggestion(sample_id)
            return {
                "sample_id": sample_id,
                "saved": saved is not None,
                "annotation": saved or self.project.annotation_template(sample_id),
                "suggestion": suggestion,
            }

    def save_annotation(
        self, sample_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        with self._lock:
            path = self.project.save_annotation(sample_id, payload)
        return {
            "saved": str(path),
            "sample_id": sample_id,
            "project": self.project_state(),
        }

    def export(self, payload: dict[str, object]) -> dict[str, object]:
        version = str(payload.get("dataset_version", "")).strip()
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            raise ValueError("dataset_version must use semantic versioning")
        raw_output = payload.get("output")
        with self._lock:
            path, report = self.project.export_manifest(
                dataset_version=version,
                output=None if raw_output in (None, "") else str(raw_output),
            )
        return {"manifest": str(path), "validation": report}

    def frame_path(self, sample_id: str) -> Path:
        record = self.project.record(sample_id)
        path = (self.project.root / record.frame_path).resolve()
        if not path.is_relative_to(self.project.root) or not path.is_file():
            raise ValueError("Frame path is outside the annotation project")
        return path


def create_server(
    application: AnnotatorApplication,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "APCAnnotator/0.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, error: Exception) -> None:
            self._json(status, {"error": str(error), "type": type(error).__name__})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"Request body must contain 1 to {MAX_BODY_BYTES} bytes")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request JSON must be an object")
            return payload

        def _static(self, relative: str) -> None:
            path = (ASSETS / relative).resolve()
            if not path.is_relative_to(ASSETS) or not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/api/project":
                    self._json(200, application.project_state())
                    return
                annotation_match = ANNOTATION_RE.match(path)
                if annotation_match:
                    self._json(200, application.annotation(annotation_match.group(1)))
                    return
                frame_match = FRAME_RE.match(path)
                if frame_match:
                    frame = application.frame_path(frame_match.group(1))
                    body = frame.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "private, max-age=3600")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path in ("/", "/index.html"):
                    self._static("index.html")
                    return
                if path in ("/app.js", "/styles.css"):
                    self._static(path[1:])
                    return
                self.send_error(404)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self._error(422, error)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._body()
                if path == "/api/import":
                    self._json(200, application.import_frames(payload))
                    return
                if path == "/api/export":
                    self._json(200, application.export(payload))
                    return
                annotation_match = ANNOTATION_RE.match(path)
                if annotation_match:
                    self._json(
                        200,
                        application.save_annotation(
                            annotation_match.group(1), payload
                        ),
                    )
                    return
                self.send_error(404)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                self._error(422, error)

    return ThreadingHTTPServer((host, port), Handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run APC's local visual annotation workbench.")
    parser.add_argument("project", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    application = AnnotatorApplication(AnnotationProject(args.project))
    server = create_server(application, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"APC annotation workbench: {url}")
    if args.open:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
