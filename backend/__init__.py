__all__ = ['make_flux_coupler_device', 'FluxPulse', 'FluxSegment']

def __getattr__(name):
    if name in __all__:
        from . import flux_coupler
        value = getattr(flux_coupler, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
