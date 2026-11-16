import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time


def main():
    parser = argparse.ArgumentParser(prog='qcalgym')
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument('--track', choices=['troubleshoot', 'tune'])
    server = commands.add_parser('serve')
    server.add_argument('--task', required=True)
    server.add_argument('--seed', type=int, default=0)
    server.add_argument('--host', default='0.0.0.0')
    server.add_argument('--port', type=int, default=8000)
    server.add_argument('--public-url')
    server.add_argument('--output', type=Path, default=Path('/runs'))
    server.add_argument('--experiments', type=int)
    server.add_argument('--shots', type=int)
    server.add_argument('--seconds', type=float)
    report = commands.add_parser('report')
    report.add_argument('directory', type=Path)
    report.add_argument('--k', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'list':
        from tasks import load_tasks
        print(json.dumps([dict(id=t.id, track=t.track, scene=t.scene, tier=t.tier) for t in load_tasks(args.track)], indent=2))
    elif args.command == 'report':
        from .scoring import aggregate
        rows = [json.loads(p.read_text()) for p in args.directory.rglob('result.json')]
        print(json.dumps(aggregate(rows, args.k), indent=2))
    else:
        run(args)


def run(args):
    from dataclasses import asdict
    from uuid import uuid4
    from tasks import get_task, build
    from backend.performance import evaluate
    from backend.benchmark.session import write_json
    from backend.benchmark.service import serve
    from .session import create_session
    from .scoring import recovery
    task = get_task(args.task)
    for name in ('experiments', 'shots', 'seconds'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f'{name} must be positive')
    output = args.output / f'{task.id}-seed{args.seed}-{uuid4().hex[:8]}'
    output.mkdir(parents=True)
    print(f'Initializing {task.id}', flush=True)
    started = time.monotonic()
    env = build(task, args.seed)
    initial = evaluate(env, task.family, unknown=task.cold)
    write_json(output / 'initial.json', initial)
    session = create_session(output / 'host', task, env, initial, args.seed, args.experiments, args.shots)
    write_json(output / 'config.json', dict(task_id=task.id, track=task.track, seed=args.seed,
               experiment_limit=session.experiment_limit, shot_limit=session.budget.max_shots))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with serve(session, host=args.host, port=args.port, public_url=args.public_url) as connection:
        write_json(output / 'connection.json', connection)
        write_json(output / 'task.json', session.objective)
        print(json.dumps(dict(connection=connection, output=str(output))), flush=True)
        deadline = time.monotonic() + args.seconds if args.seconds else None
        while not stop.wait(.2):
            if session.submission is not None or (deadline is not None and time.monotonic() >= deadline):
                break
    print('Scoring final controls', flush=True)
    final = session.final_performance()
    if not final['complete']:
        write_json(output / 'final.json', final)
        raise RuntimeError('Final performance calculation did not complete')
    write_json(output / 'final.json', final)
    write_json(output / 'calibration.json', asdict(session.env.cal))
    result = dict(task_id=task.id, track=task.track, tier=task.tier, seed=args.seed,
                  passed=final['passed'], recovery_score=recovery(initial, final, task.cold),
                  used=session.status()['used'], seconds=time.monotonic() - started)
    write_json(output / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
