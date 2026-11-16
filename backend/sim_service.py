import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import fcntl
import json
from multiprocessing import get_context
import os
from pathlib import Path
import select
import socket
import socketserver
import threading
import time
from .sim_engine import digest, solve_job
MAX_REQUEST = 64 * 1024 ** 2

class Scheduler:

    def __init__(self, workers):
        self.workers = workers
        self.lock = threading.Condition()
        self.queue = deque()
        self.slots = [dict(pool=self.new_pool(), busy=False, affinity=None) for _ in range(workers)]
        self.completed = 0
        self.prefix_hits = 0
        self.cancelled_points = 0
        self.last_activity = time.monotonic()
        self.stopping = False
        self.thread = threading.Thread(target=self.dispatch, daemon=True)
        self.thread.start()

    @staticmethod
    def new_pool():
        return ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'))

    def submit(self, spec, jobs):
        request = dict(spec=spec, key=digest(spec), jobs=deque(enumerate(jobs)), results=deque(), active=0, cancelled=False)
        with self.lock:
            if self.stopping:
                raise RuntimeError('Simulation service is stopping')
            self.queue.append(request)
            self.last_activity = time.monotonic()
            self.lock.notify_all()
        return request

    def cancel(self, request):
        with self.lock:
            request['cancelled'] = True
            self.cancelled_points += len(request['jobs'])
            request['jobs'].clear()
            self.lock.notify_all()

    def dispatch(self):
        while True:
            with self.lock:
                self.queue = deque((r for r in self.queue if r['jobs'] and (not r['cancelled'])))
                idle = [s for s in self.slots if not s['busy']]
                if self.stopping:
                    return
                if not self.queue or not idle:
                    self.lock.wait(0.2)
                    continue
                request = self.queue.popleft()
                slot = next((s for s in idle if s['affinity'] == request['key']), idle[0])
                index, job = request['jobs'].popleft()
                slot['busy'] = True
                slot['affinity'] = request['key']
                request['active'] += 1
                if request['jobs']:
                    self.queue.append(request)
            try:
                future = slot['pool'].submit(solve_job, request['spec'], job)
                future.add_done_callback(lambda f, s=slot, r=request, i=index: self.finished(s, r, i, f))
            except Exception as exc:
                self.finished(slot, request, index, None, exc)

    def finished(self, slot, request, index, future, error=None):
        try:
            if error:
                raise error
            value = future.result()
            result = dict(index=index, value=value)
        except Exception as exc:
            result = dict(index=index, error=f'{type(exc).__name__}: {exc}')
            if isinstance(exc, BrokenProcessPool):
                slot['pool'].shutdown(wait=False, cancel_futures=True)
                slot['pool'] = self.new_pool()
                slot['affinity'] = None
        with self.lock:
            slot['busy'] = False
            request['active'] -= 1
            if not request['cancelled']:
                request['results'].append(result)
            self.completed += 1
            self.prefix_hits += result.get('value', {}).get('stats', {}).get('prefix_hits', 0)
            self.last_activity = time.monotonic()
            self.lock.notify_all()

    def status(self):
        with self.lock:
            return dict(pid=os.getpid(), workers=self.workers, busy=sum((s['busy'] for s in self.slots)), queued_points=sum((len(r['jobs']) for r in self.queue)), completed_points=self.completed, prefix_hits=self.prefix_hits, cancelled_points=self.cancelled_points)

    def close(self):
        with self.lock:
            self.stopping = True
            for request in self.queue:
                request['cancelled'] = True
                request['jobs'].clear()
            self.lock.notify_all()
        self.thread.join()
        for slot in self.slots:
            slot['pool'].shutdown(wait=True, cancel_futures=True)

class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    request_queue_size = 64

class Handler(socketserver.StreamRequestHandler):

    def send(self, message):
        self.wfile.write(json.dumps(message, separators=(',', ':')).encode() + b'\n')
        self.wfile.flush()

    def disconnected(self):
        readable, _, _ = select.select([self.connection], [], [], 0)
        return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b''

    def handle(self):
        request = None
        scheduler = self.server.scheduler
        try:
            self.connection.settimeout(30)
            raw = self.rfile.readline(MAX_REQUEST + 1)
            if len(raw) > MAX_REQUEST or not raw.endswith(b'\n'):
                raise ValueError('Simulation request too large or incomplete')
            message = json.loads(raw)
            self.connection.settimeout(None)
            if message.get('op') == 'status':
                self.send(scheduler.status())
                return
            if message.get('op') == 'stop':
                self.send({'stopping': True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if message.get('op') != 'solve':
                raise ValueError('Unknown simulation operation')
            jobs = message['jobs']
            if not isinstance(jobs, list) or not 1 <= len(jobs) <= 1024:
                raise ValueError('Expected 1 to 1024 simulation points')
            request = scheduler.submit(message['spec'], jobs)
            sent = 0
            while sent < len(jobs):
                if self.disconnected() or scheduler.stopping:
                    return
                with scheduler.lock:
                    results = list(request['results'])
                    request['results'].clear()
                    if not results:
                        scheduler.lock.wait(0.2)
                for result in results:
                    self.send(result)
                    sent += 1
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as exc:
            try:
                self.send({'error': f'{type(exc).__name__}: {exc}'})
            except OSError:
                pass
        finally:
            if request is not None:
                scheduler.cancel(request)

def serve(path, workers, idle_seconds=900):
    path = Path(path)
    path.parent.mkdir(mode=448, parents=True, exist_ok=True)
    os.umask(63)
    with path.with_suffix('.server.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path.unlink(missing_ok=True)
        server = Server(str(path), Handler)
        server.scheduler = Scheduler(workers)
        os.chmod(path, 384)

        def reap_idle():
            while not server.scheduler.stopping:
                time.sleep(2)
                status = server.scheduler.status()
                if not status['busy'] and (not status['queued_points']) and (time.monotonic() - server.scheduler.last_activity > idle_seconds):
                    server.shutdown()
                    return
        threading.Thread(target=reap_idle, daemon=True).start()
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            server.scheduler.close()
            server.server_close()
            path.unlink(missing_ok=True)

def main():
    from .sim_client import socket_path, control
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default=str(socket_path()))
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--idle-seconds', type=float, default=900)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    if args.status or args.stop:
        print(json.dumps(control(args.socket, 'stop' if args.stop else 'status'), indent=2))
    else:
        if not 1 <= args.workers <= 64:
            parser.error('workers must be between 1 and 64')
        serve(args.socket, args.workers, args.idle_seconds)
if __name__ == '__main__':
    main()
