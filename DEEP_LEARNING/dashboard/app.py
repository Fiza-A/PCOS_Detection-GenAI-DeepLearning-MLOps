"""Small local-only HTML inference dashboard; does not train or save models."""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "DEEP_LEARNING" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from predict import PCOSPredictor  # noqa: E402
MLOPS_SRC = ROOT / "MLOPS" / "src"
if str(MLOPS_SRC) not in sys.path:
    sys.path.insert(0, str(MLOPS_SRC))
from prediction_logging import current_model_version, log_prediction  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000


class DashboardHandler(BaseHTTPRequestHandler):
    predictor: PCOSPredictor

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/schema":
            body = json.dumps(self.predictor.schema()).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        file_map = {"/": ("index.html", "text/html; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if path not in file_map:
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        filename, content_type = file_map[path]
        self._send(200, content_type, (DASHBOARD_DIR / filename).read_bytes())

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/predict":
            self._send(404, "application/json; charset=utf-8", b'{"error":"Not found"}')
            return
        started = time.perf_counter()
        model_version = current_model_version()

        def log_outcome(status: str, result: dict | None = None, error_category: str | None = None) -> None:
            try:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if status == "SUCCESS" and result is not None:
                    log_prediction(
                        model_version=model_version,
                        status=status,
                        inference_duration_ms=elapsed_ms,
                        predicted_label=result["predicted_label"],
                        probability=result["probability"],
                    )
                else:
                    log_prediction(
                        model_version=model_version,
                        status="FAILURE",
                        inference_duration_ms=elapsed_ms,
                        error_category=error_category,
                    )
            except Exception as logging_error:
                # A local monitoring write failure must not change inference responses.
                print(f"MLOps prediction logging unavailable: {type(logging_error).__name__}")

        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 64_000:
                raise ValueError("Request body is empty or too large.")
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Submit a JSON object containing the displayed features.")
            result = self.predictor.predict(payload)
            log_outcome("SUCCESS", result=result)
            self._send(200, "application/json; charset=utf-8", json.dumps(result).encode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            log_outcome("FAILURE", error_category="input_validation")
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, "application/json; charset=utf-8", body)
        except Exception:
            log_outcome("FAILURE", error_category="inference_error")
            # Do not log submitted feature values; keep a server-side traceback for diagnostics.
            import traceback
            traceback.print_exc()
            self._send(500, "application/json; charset=utf-8", b'{"error":"Prediction could not be completed. Check the local server output."}')

    def log_message(self, format: str, *args: object) -> None:
        # Requests are logged without query/body data; never log patient features.
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    predictor = PCOSPredictor()
    DashboardHandler.predictor = predictor
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print("PCOS-MLOps — Deep Learning Prediction")
    print(f"Frozen checkpoint: {predictor.checkpoint_payload['experiment']} (evaluation mode)")
    print(f"Local dashboard: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

