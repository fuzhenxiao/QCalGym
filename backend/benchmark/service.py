from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import secrets
import threading
from urllib.parse import unquote
from .data_access import participant_response

@contextmanager
def serve(session, progress=None, host='127.0.0.1', port=0, public_url=None):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def authorized(self):
            return secrets.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)

        def send(self, status, body, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.authorized():
                return self.send(401, b'{"error":"unauthorized"}')
            try:
                if self.path != '/call':
                    raise ValueError('Unknown endpoint')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1000000:
                    raise ValueError('Invalid request size')

                def invalid_constant(value):
                    raise ValueError('Nonfinite JSON number')
                request = json.loads(self.rfile.read(length), parse_constant=invalid_constant)
                if progress:
                    progress(stage=request['action'], experiment=(request.get('params') or {}).get('name'), wait_until=None)
                try:
                    result = session.dispatch(**request)
                finally:
                    if progress:
                        progress(stage='agent_tools', experiment=None, experiments=session.experiment_count, shots=session.env.ledger.shots)
                result = participant_response(request['action'], result)
                if callable(getattr(session, 'budget_notice', None)):
                    session.budget_notice()
                self.send(200, json.dumps(result, allow_nan=False).encode())
            except Exception as exc:
                self.send(400, json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}).encode())

        def do_GET(self):
            if self.path == '/health':
                return self.send(200, b'{"status":"ready"}')
            if not self.authorized():
                return self.send(401, b'{"error":"unauthorized"}')
            try:
                if not self.path.startswith('/artifacts/'):
                    raise ValueError('Unknown endpoint')
                path = session.artifact(unquote(self.path[len('/artifacts/'):]))
                data = path.read_bytes()
                self.send(200, data, mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            except ValueError as exc:
                self.send(404, json.dumps({'error': str(exc)}).encode())
    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {'url': public_url or f'http://127.0.0.1:{server.server_port}', 'token': token}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
