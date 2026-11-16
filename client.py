import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

class Client:

    def __init__(self, config=None):
        self.config = config or json.loads(Path(os.environ['QCALIBENCH_CONNECTION']).read_text())

    def call(self, action, params=None, request_id=None):
        body = {'action': action, 'params': params or {}, 'request_id': request_id or uuid4().hex}
        req = Request(self.config['url'] + '/call', data=json.dumps(body, allow_nan=False).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.config['token']})
        try:
            with urlopen(req, timeout=None) as response:
                return json.load(response)
        except HTTPError as exc:
            return json.load(exc)

    def run_experiment(self, name, parameters, request_id=None):
        return self.call('run_experiment', {'name': name, 'parameters': parameters}, request_id)

    def download(self, artifact):
        req = Request(self.config['url'] + '/artifacts/' + quote(artifact), headers={'Authorization': 'Bearer ' + self.config['token']})
        with urlopen(req, timeout=None) as response:
            return response.read()
TOOLS = [{'type': 'function',
  'function': {'name': 'controls',
               'description': 'List all writable calibration fields, exact value schemas, index '
                              'conventions and coupled constraints.',
               'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'catalog',
               'description': 'List all predefined experiments and parameter schemas.',
               'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'status',
               'description': 'Read current calibration, public objective and remaining budget.',
               'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'run_experiment',
               'description': 'Execute one predefined experiment; obtain fit summaries, figure '
                              'artifacts and data access handles. Full scan arrays and raw IQ are '
                              'available on demand via check_raw_data or artifact download.',
               'parameters': {'type': 'object',
                              'properties': {'name': {'type': 'string'},
                                             'parameters': {'type': 'object'}},
                              'required': ['name', 'parameters'],
                              'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'update_calibration',
               'description': 'Explicitly update calibration fields; does not modify the device '
                              'model.',
               'parameters': {'type': 'object',
                              'properties': {'updates': {'type': 'array',
                                                         'items': {'type': 'object',
                                                                   'properties': {'field': {'type': 'string'},
                                                                                  'index': {'oneOf': [{'type': 'integer'},
                                                                                                      {'type': 'array',
                                                                                                       'items': {'type': 'integer'}}]},
                                                                                  'value': {}},
                                                                   'required': ['field',
                                                                                'index',
                                                                                'value'],
                                                                   'additionalProperties': False}}},
                              'required': ['updates'],
                              'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'check_raw_data',
               'description': 'Read stored public experiment data without acquiring again. Default '
                              'manifest lists point names and paginated raw-array/acquisition '
                              'metadata. points/raw reads a named array slice. '
                              'fit/meta/state/response explicitly reads a complete recorded '
                              'section (response can be large). Full JSON and NPZ artifacts are '
                              'also downloadable. Available after submission.',
               'parameters': {'type': 'object',
                              'properties': {'request_id': {'type': 'string'},
                                             'section': {'type': 'string',
                                                         'enum': ['manifest',
                                                                  'points',
                                                                  'raw',
                                                                  'fit',
                                                                  'meta',
                                                                  'state',
                                                                  'response'],
                                                         'default': 'manifest'},
                                             'array': {'type': ['string', 'null'], 'default': None},
                                             'start': {'type': 'integer',
                                                       'minimum': 0,
                                                       'default': 0},
                                             'stop': {'type': 'integer',
                                                      'minimum': 1,
                                                      'default': 100}},
                              'required': ['request_id'],
                              'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'get_data',
               'description': 'Read up to 2000 samples of an acquired raw IQ array; complex values '
                              'use {real, imag}. Discover names using check_raw_data first.',
               'parameters': {'type': 'object',
                              'properties': {'request_id': {'type': 'string'},
                                             'array': {'type': 'string'},
                                             'start': {'type': 'integer', 'minimum': 0},
                                             'stop': {'type': 'integer', 'minimum': 1}},
                              'required': ['request_id', 'array'],
                              'additionalProperties': False}}},
 {'type': 'function',
  'function': {'name': 'submit',
               'description': 'Finish this episode with a report and claims; receipt does not '
                              'imply benchmark success.',
               'parameters': {'type': 'object',
                              'properties': {'report': {'type': 'string'},
                                             'claims': {'type': 'object'}},
                              'required': ['report'],
                              'additionalProperties': False}}}]


if __name__ == "__main__":
    print(json.dumps(TOOLS, indent=2))
