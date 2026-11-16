from copy import deepcopy
VERSION = '1.1-on-demand-data'

def describe(value):
    if isinstance(value, list):
        shape = []
        cursor = value
        while isinstance(cursor, list):
            shape.append(len(cursor))
            cursor = cursor[0] if cursor else None
        return {'shape': shape, 'length': len(value)}
    return {'type': type(value).__name__}

def compact(value):
    if isinstance(value, list):
        if len(value) > 8:
            return {'stored': True, **describe(value)}
        return [compact(v) for v in value]
    if isinstance(value, dict):
        if len(value) > 32:
            return {'stored': True, 'entries': len(value)}
        return {k: compact(v) for k, v in value.items()}
    return value

def participant_response(action, response):
    if action not in {'run_experiment', 'update_calibration', 'submit'}:
        return response
    result = {k: compact(v) for k, v in response.items() if k not in {'points', 'raw_data', 'fit', 'state'}}
    result['api_version'] = VERSION
    if 'points' in response:
        result['point_arrays'] = {k: describe(v) for k, v in response['points'].items()}
    if 'fit' in response:
        result['fit'] = compact({k: v for k, v in response['fit'].items() if k != 'yfit'})
        if 'yfit' in response['fit']:
            result['fit_curve'] = describe(response['fit']['yfit'])
    if 'state' in response:
        result['state'] = {k: deepcopy(v) for k, v in response['state'].items() if k in {'revision', 'used', 'budget', 'submitted'}}
    raw = response.get('raw_data')
    if raw is not None:
        result['raw_data'] = {'artifact': raw['artifact'], 'kind': raw['kind'], 'array_count': len(raw['arrays']), 'acquisition_count': len(raw['acquisitions'])}
    if response.get('request_id'):
        result['data_access'] = {'request_id': response['request_id'], 'response_artifact': response.get('response_artifact'), 'instruction': "Full measured data are available, not included in this summary. Call check_raw_data(request_id, section='manifest') for array names and metadata; section='points' or 'raw', array=NAME, start=0, stop=100 reads a slice. section='fit' or 'state' reads the full recorded section. Download response_artifact for the complete public JSON or raw_data.artifact for the complete NPZ using Client.download(artifact). Use status for current calibration."}
    return result
