from copy import deepcopy
from .experiment_budget import BudgetExhausted
FIELDS = {'readout_spectrum': {'readout': {'readout_freq'}}, 'qubit_spectrum': {}, 'rabi': {}, 'iq': {}, 'ramsey': {}, 'drag': {}, 'coherence': {}, 'readout_power': {'readout': {'readout_amplitude'}}, 'readout_duration': {'readout': {'readout_duration'}}, 'flux_spectroscopy': {'flux': {'qubit_bias', 'coupler_bias'}}, 'native_phase': {'native': {'frequency', 'duration', 'amplitudes'}}, 'flux_ramsey': {'flux': {'qubit_bias', 'coupler_bias', 'mix', 'inverse_tau', 'inverse_tail'}}}

def run_scan(session, kind, requests, shots=1024):
    if kind not in FIELDS or not isinstance(requests, list) or (not 1 <= len(requests) <= 441):
        raise ValueError('Unknown scan family or more than 441 acquisitions')
    if type(shots) is not int or not 16 <= shots <= 20000:
        raise ValueError('Scan shots outside 16..20000')
    requests = deepcopy(requests)
    saved = session._env.cal.copy()
    allowed = FIELDS[kind]
    methods = {'readout': session.set_readout_drive_controls, 'flux': session.set_flux_controls, 'native': session.set_native_gate}
    axes = None
    try:
        for r in requests:
            if set(r) - {'blocks', 'preparation', 'basis', 'analysis_phases', 'controls'}:
                raise ValueError('Unknown scan request field')
            controls = r.get('controls', {})
            shape = {group: set(updates) for group, updates in controls.items()}
            if axes is not None and shape != axes:
                raise ValueError('Scan control axes must remain fixed')
            axes = shape
            if set(controls) - set(allowed):
                raise ValueError('Control group outside scan family')
            session._env.cal = saved.copy()
            for group, updates in controls.items():
                if set(updates) - allowed[group]:
                    raise ValueError('Control outside scan family')
                if group == 'native' and len(updates) > 1:
                    raise ValueError('One native gate axis per scan')
                methods[group](**updates)
            blocks = r.get('blocks', [])
            if kind.startswith('readout') or kind == 'iq':
                if blocks:
                    raise ValueError('Readout scans have no drive sequence')
            elif kind in ('ramsey', 'coherence'):
                if any((isinstance(b, dict) for b in blocks)):
                    raise ValueError('Coherence scan accepts delays only')
            elif kind == 'flux_ramsey':
                if any((isinstance(b, dict) and set(b) != {'duration', 'pulses'} for b in blocks)):
                    raise ValueError('Flux Ramsey accepts Z-pulse blocks and waits only')
            elif kind == 'native_phase':
                if blocks not in ([], [{'gate': 'cz', 'repetitions': 1}]):
                    raise ValueError('Native phase scan measures one CZ and its reference')
            elif any((not isinstance(b, dict) or set(b) != {'xy'} for b in blocks)):
                raise ValueError('This family accepts XY probes only')
    finally:
        session._env.cal = saved
        session._env.sync_physics()
    used = session._env.ledger
    if session._max_acquisitions is not None and used.experiments + len(requests) > session._max_acquisitions:
        raise BudgetExhausted('Acquisition budget exhausted before scan')
    if session._max_shots is not None and used.shots + shots * len(requests) > session._max_shots:
        raise BudgetExhausted('Shot budget exhausted before scan')
    session._admit_experiment()
    before = used.shots
    session._in_scan = True
    results = []
    session._record_update('standard_scan_start', dict(kind=kind, requests=requests, shots=shots))
    try:
        with session._env.sim.scan_cache():
            for r in requests:
                session._env.cal = saved.copy()
                for group, updates in r.pop('controls', {}).items():
                    methods[group](**updates)
                results.append(session.acquire(**{'blocks': [], **r}, shots=shots))
        return results
    finally:
        session._in_scan = False
        if used.shots > before:
            session.experiment_count += 1
        session._env.cal = saved
        session._env.sync_physics()
        session._record_update('standard_scan_end', dict(kind=kind, completed=len(results), requested=len(requests), experiments=session.experiment_count))
