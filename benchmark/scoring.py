import math
from collections import defaultdict


def gap(metric):
    error, threshold = metric.get('error'), metric['threshold']
    if error is None:
        return 1.0
    if error <= threshold:
        return 0.0
    return min(1.0, math.log(error / threshold) / math.log(0.5 / threshold))


def recovery(initial, final, cold=False):
    start = len(initial['metrics']) if cold else sum(map(gap, initial['metrics']))
    end = sum(map(gap, final['metrics']))
    return max(0.0, 1 - end / start) if start else float(end == 0)


def aggregate(results, k=1):
    groups = defaultdict(list)
    for result in results:
        groups[result['task_id']].append(result)
    if not groups or any(len(rows) < k for rows in groups.values()):
        raise ValueError(f'At least {k} completed runs per task are required')
    values = []
    for rows in groups.values():
        n, c = len(rows), sum(r['passed'] for r in rows)
        values.append(1 - math.comb(n - c, k) / math.comb(n, k))
    return {'tasks': len(groups), 'runs': len(results), f'pass@{k}': sum(values) / len(values),
            'recovery_score': sum(sum(r['recovery_score'] for r in rows) / len(rows) for rows in groups.values()) / len(groups)}
