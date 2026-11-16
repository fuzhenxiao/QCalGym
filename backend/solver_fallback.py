import functools
import qutip
_METHODS = {'dop853', 'vern7', 'vern9', 'adams'}

def _robust(solver):
    if getattr(solver, '_qcal_fallback', False):
        return solver

    @functools.wraps(solver)
    def call(*args, **kwargs):
        try:
            return solver(*args, **kwargs)
        except Exception as exc:
            options = kwargs.get('options') or {}
            if 'stiff' not in str(exc).lower() or options.get('method') not in _METHODS:
                raise
            retry = {k: v for k, v in options.items() if k in ('nsteps', 'atol', 'rtol', 'store_states', 'store_final_state', 'normalize_output', 'progress_bar', 'max_step', 'min_step')}
            retry['method'] = 'lsoda'
            retry['nsteps'] = max(int(options.get('nsteps', 1000)), 1000)
            return solver(*args, **{**kwargs, 'options': retry})
    call._qcal_fallback = True
    return call
qutip.sesolve = _robust(qutip.sesolve)
qutip.mesolve = _robust(qutip.mesolve)
