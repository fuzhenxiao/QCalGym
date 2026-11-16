REASONING_MODEL = {
    'base_url': 'https://integrate.api.nvidia.com/v1',
    'model': 'nvidia/nemotron-3-nano-30b-a3b',
    'api_key': '',
}
VISION_MODEL = {
    'base_url': 'https://integrate.api.nvidia.com/v1',
    'model': 'nvidia/ising-calibration-1-35b-a3b',
    'api_key': '',
}

import argparse
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import venv
import zipfile

from client import Client, TOOLS

UPSTREAM = 'https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint'
REVISION = 'd8d37bc4de28d2347d8e21405903b29a2891f793'
CONSTRAINTS = '''langchain==1.2.15
langchain-core==1.3.0
langchain-openai==1.2.0
langchain-anthropic==1.4.1
langgraph==1.1.9
langgraph-api==0.13.4
langgraph-sdk==0.3.15
langgraph-cli==0.4.31
langgraph-checkpoint==4.2.0
langgraph-checkpoint-sqlite==3.1.1
langgraph-prebuilt==1.0.10
langgraph-runtime-inmem==0.33.4
deepagents-acp==0.0.11
'''

BRIDGE = '''import base64
import json
from pathlib import Path
from qcalgym_client import Client

def invoke(action, params):
    root = Path(__file__).resolve().parents[1]
    client = Client(json.loads((root / 'qcalgym_connection.json').read_text()))
    response = client.call(action, params)
    plots, arrays = [], {}
    if response.get('status') == 'success' and action == 'run_experiment':
        for figure in response.get('figures', []):
            if figure.get('format') == 'png':
                plots.append({'name': figure.get('name', 'measurement'), 'format': 'png',
                              'data': base64.b64encode(client.download(figure['artifact'])).decode()})
        stored = response
        if response.get('response_artifact'):
            stored = json.loads(client.download(response['response_artifact']))
        for name, values in stored.get('points', {}).items():
            if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                arrays[name] = values
        values = stored.get('fit', {}).get('yfit')
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            arrays['fit_yfit'] = values
    result = {'status': response.get('status', 'failed'), 'data': {'benchmark': response},
              'arrays': arrays, 'plots': plots}
    if result['status'] != 'success':
        result['status'] = 'failed'
        result['error'] = response.get('error', response.get('error_type', 'Benchmark call failed'))
    return result
'''

LAUNCHER = '''import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

def model_settings():
    return json.loads((ROOT / 'qcalgym_models.json').read_text())

def model_key(role, settings):
    variable = 'QCALGYM_' + role.upper() + '_API_KEY'
    key = settings['api_key'] or os.environ.get(variable, '')
    if not key:
        raise ValueError(f'Set {role} api_key at the top of setup_blueprint.py or set {variable}.')
    return key

def create_reasoning_model(settings):
    from langchain_openai import ChatOpenAI

    class RoutedChat(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            raw = response if isinstance(response, dict) else response.model_dump()
            for generation, choice in zip(result.generations, raw.get('choices', [])):
                for field in ('reasoning_content', 'reasoning_details'):
                    if field in choice.get('message', {}):
                        generation.message.additional_kwargs[field] = choice['message'][field]
            return result

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            for message, raw in zip(self._convert_input(input_).to_messages(), payload['messages']):
                if raw.get('role') == 'assistant':
                    for field in ('reasoning_content', 'reasoning_details'):
                        if field in message.additional_kwargs:
                            raw[field] = message.additional_kwargs[field]
            return payload

    return RoutedChat(model=settings['model'], base_url=settings['base_url'],
                      api_key=model_key('reasoning', settings), use_responses_api=False,
                      disable_streaming=True, timeout=None)

def configure_vision(settings):
    import yaml
    path = ROOT / 'config.yaml'
    config = yaml.safe_load(path.read_text()) or {}
    config['vlm'] = {'provider': 'custom', 'model': settings['model'],
                     'endpoint': settings['base_url'] + '/chat/completions',
                     'api_key_env': 'QCALGYM_VISION_API_KEY',
                     'temperature': 0, 'max_tokens': 8192}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

def check():
    from core.discovery import discover_experiments
    from core.runner import run_experiment
    expected = set(json.loads((ROOT / 'qcalgym_schemas.json').read_text()))
    found = {item.name for item in discover_experiments(ROOT / 'scripts')}
    if found != expected:
        raise RuntimeError(f'Experiment discovery mismatch: {found ^ expected}')
    response = run_experiment('qcal_status', {'verbose': False}, ROOT / 'scripts',
                              python_path=sys.executable)
    if response['status'] != 'success':
        raise RuntimeError(response.get('error', 'Status query failed'))
    print(f'Connected: {len(found)} QCalGym functions discovered; status query passed.')

async def run(settings):
    from langchain_core.messages import HumanMessage
    import server
    from qcalgym_client import Client
    reasoning = create_reasoning_model(settings['reasoning'])
    os.environ['QCALGYM_VISION_API_KEY'] = model_key('vision', settings['vision'])
    server.create_chat_model = lambda name: reasoning
    connection = json.loads((ROOT / 'qcalgym_connection.json').read_text())
    state = Client(connection).call('status')
    if state.get('status') != 'success':
        raise RuntimeError(state)
    prompt = ('Calibrate the connected QCalGym device. Read qcal_catalog and qcal_controls '
              'for experiment and update schemas. Use measurements to diagnose and repair '
              'the device, then finish with qcal_submit. The initial performance table is '
              'fixed; use experiments for feedback after updates. Respect the task budget '
              'and, for Tune, change only the listed parameters_to_fix. '
              'All benchmark experiments, control updates and submission are authorized. '
              'Use only the public QCalGym interface for device information. Initial state: '
              + json.dumps(state))
    agent, backend, checkpointer = server.create_agent()
    config = {'configurable': {'thread_id': 'qcalgym'}, 'recursion_limit': 1000}
    with (ROOT / 'qcalgym_agent_events.jsonl').open('a') as stream:
        async for event in agent.astream_events({'messages': [HumanMessage(content=prompt)]},
                                                config=config, version='v2'):
            if event['event'] in ('on_tool_start', 'on_tool_end', 'on_chat_model_end'):
                record = {'event': event['event'], 'name': event.get('name'), 'data': event.get('data')}
                stream.write(json.dumps(record, default=str) + '\\n')
                stream.flush()
                print(event['event'] + ': ' + event.get('name', ''), flush=True)

def main():
    parser = argparse.ArgumentParser(description='Run Blueprint against the connected QCalGym task.')
    parser.add_argument('--check', action='store_true', help='Check discovery and HTTP access without a model call.')
    parser.add_argument('--model', help='Override the configured reasoning model ID for this launch.')
    args = parser.parse_args()
    settings = model_settings()
    if args.model:
        settings['reasoning']['model'] = args.model
    configure_vision(settings['vision'])
    check()
    if not args.check:
        asyncio.run(run(settings))

if __name__ == '__main__':
    main()
'''

INSTRUCTIONS = '''

## QCalGym connection

This workspace is connected to a running QCalGym task. Use the registered
qcal_ functions through run_experiment(experiment_name="qcal_NAME", params={...}).
Start with qcal_status, qcal_catalog and qcal_controls. qcal_catalog returns
the full experiment parameter schemas; qcal_controls returns writable fields,
indices and bounds. A dedicated qcal_NAME function exists for every available
experiment. qcal_run_experiment(name=NAME, parameters={...}) is also available.
Apply controls with qcal_update_calibration and finish with qcal_submit.
The initial performance table is fixed, so use experiments for feedback.
Experiment plots and numeric arrays are stored in native Blueprint history
and are available to lab and vlm_inspect. Raw data remain accessible through
qcal_check_raw_data and qcal_get_data. Exact schemas are also saved under
data/knowledge/documents/qcalgym_catalog.json and in qcalgym_tools.json.
'''


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def configured_models():
    models = {}
    for role, settings in (('reasoning', REASONING_MODEL), ('vision', VISION_MODEL)):
        values = dict(settings)
        values['base_url'] = values['base_url'].rstrip('/')
        suffix = '/chat/completions'
        if values['base_url'].endswith(suffix):
            values['base_url'] = values['base_url'][:-len(suffix)]
        if not values['base_url'].startswith(('http://', 'https://')) or not values['model']:
            raise ValueError(f'Set the {role} base_url and model at the top of this script.')
        models[role] = values
    return models


def download_source(directory):
    archive = directory / 'blueprint.zip'
    url = f'https://codeload.github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint/zip/{REVISION}'
    print(f'Downloading Blueprint {REVISION[:12]}...', flush=True)
    with urlopen(url, timeout=60) as response, archive.open('wb') as output:
        shutil.copyfileobj(response, output)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in item.filename:
                raise ValueError('Invalid path in Blueprint archive')
        bundle.extractall(directory / 'source')
    return directory / 'source' / f'Quantum-Calibration-Agent-Blueprint-{REVISION}'


def prepare_source(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ('core', 'tools', 'vlm', 'data/knowledge'):
        shutil.copytree(source / name, output / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('server.py', 'prompt.py', 'cli.py', 'config.yaml', 'pyproject.toml', 'LICENSE', 'ATTRIBUTIONS'):
        shutil.copy2(source / name, output / name)
    if (source / '.env.example').exists():
        shutil.copy2(source / '.env.example', output / '.env.example')
    runner = output / 'core/runner.py'
    content = runner.read_text(encoding='utf-8')
    content = content.replace('            stdout_data = []\n',
        '            stdout_data = []\n'
        '            import threading\n'
        '            stdout_reader = threading.Thread(target=lambda: stdout_data.append(process.stdout.read()), daemon=True)\n'
        '            stdout_reader.start()\n', 1)
    content = content.replace('            stdout_content = process.stdout.read()\n',
        '            stdout_reader.join()\n            stdout_content = "".join(stdout_data)\n', 1)
    runner.write_text(content, encoding='utf-8')
    lab = output / 'tools/lab_tool.py'
    content = lab.read_text(encoding='utf-8')
    old = 'exp_id = f"{now.strftime(\'%Y%m%d_%H%M%S\')}_{experiment_name}"'
    lab.write_text(content.replace(old, old + ' + "_" + __import__("uuid").uuid4().hex', 1), encoding='utf-8')
    storage = output / 'core/storage.py'
    content = storage.read_text(encoding='utf-8').replace(
        'filename = f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}_{result.type}.h5"',
        'filename = f"{result.id}.h5"', 1)
    storage.write_text(content, encoding='utf-8')
    prompt = output / 'data/knowledge/system-prompt.md'
    prompt.write_text(prompt.read_text(encoding='utf-8') + INSTRUCTIONS, encoding='utf-8')
    write_json(output / '.qcalgym-blueprint.json', {'upstream': UPSTREAM, 'revision': REVISION, 'functions': []})


def function_source(name, spec, action, experiment=None):
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f'Invalid function name: {name}')
    schema = spec['parameters']
    props = schema.get('properties', {})
    required = set(schema.get('required', []))
    ordered = sorted(props, key=lambda key: key not in required)
    arguments = []
    for key in ordered:
        if not key.isidentifier() or keyword.iskeyword(key):
            raise ValueError(f'Invalid parameter name: {key}')
        kind = props[key].get('type', 'object')
        if isinstance(kind, list):
            kind = next((item for item in kind if item != 'null'), 'object')
        hint = {'number': 'float', 'integer': 'int', 'array': 'list',
                'object': 'dict', 'boolean': 'bool', 'string': 'str'}[kind]
        default = '' if key in required else ' = ' + repr(props[key].get('default'))
        arguments.append(key + ': ' + hint + default)
    if not arguments:
        arguments = ['verbose: bool = False']
    params = '{' + ', '.join(repr(key) + ': ' + key for key in ordered) + '}'
    body = f'    params = {params}\n'
    body += '    params = {key: value for key, value in params.items() if value is not None}\n'
    if experiment is not None:
        body += f'    params = {{"name": {experiment!r}, "parameters": params}}\n'
    return ('from ._qcalgym_bridge import invoke\n\n'
            f'def {name}({", ".join(arguments)}) -> dict:\n'
            f'    {spec.get("description", name)!r}\n' + body + f'    return invoke({action!r}, params)\n')


def export(output, connection, catalog):
    marker = output / '.qcalgym-blueprint.json'
    manifest = json.loads(marker.read_text())
    scripts = output / 'scripts'
    scripts.mkdir(exist_ok=True)
    for name in manifest['functions']:
        (scripts / f'{name}.py').unlink(missing_ok=True)
    (scripts / '__init__.py').touch()
    (scripts / '_qcalgym_bridge.py').write_text(BRIDGE, encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('client.py'), output / 'qcalgym_client.py')
    definitions = {f'qcal_{tool["function"]["name"]}':
                   (tool['function'], tool['function']['name'], None) for tool in TOOLS}
    for name, spec in catalog.items():
        key = 'qcal_' + name
        if key in definitions:
            raise ValueError(f'Experiment name conflicts with an API action: {name}')
        definitions[key] = (spec, 'run_experiment', name)
    for name, (spec, action, experiment) in definitions.items():
        (scripts / f'{name}.py').write_text(function_source(name, spec, action, experiment), encoding='utf-8')
    write_json(output / 'qcalgym_schemas.json', {name: spec for name, (spec, _, _) in definitions.items()})
    write_json(output / 'qcalgym_tools.json', TOOLS)
    documents = output / 'data/knowledge/documents'
    documents.mkdir(parents=True, exist_ok=True)
    write_json(documents / 'qcalgym_catalog.json', catalog)
    write_json(output / 'qcalgym_connection.json', connection)
    (output / 'qcalgym_connection.json').chmod(0o600)
    write_json(output / 'qcalgym_models.json', configured_models())
    (output / 'qcalgym_models.json').chmod(0o600)
    (output / 'run_qcalgym.py').write_text(LAUNCHER, encoding='utf-8')
    (output / 'qcalgym_constraints.txt').write_text(CONSTRAINTS, encoding='utf-8')
    manifest['functions'] = list(definitions)
    write_json(marker, manifest)


def main():
    parser = argparse.ArgumentParser(description='Download Blueprint and connect its native tools to QCalGym.')
    parser.add_argument('--connection', type=Path, required=True, help='connection.json from a running QCalGym task')
    parser.add_argument('--output', type=Path, default=Path('blueprint'), help='Dedicated Blueprint workspace (default: blueprint)')
    parser.add_argument('--source', type=Path, help='Copy an existing Blueprint checkout instead of downloading')
    parser.add_argument('--skip-install', action='store_true', help='Generate the workspace without installing dependencies')
    parser.add_argument('--run', action='store_true', help='Start the agent after setup using your configured model credentials')
    parser.add_argument('--model', help='Override the configured reasoning model ID for --run')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 or newer is required')
    if args.run and args.skip_install:
        parser.error('--run cannot be combined with --skip-install')
    output = args.output.resolve()
    marker = output / '.qcalgym-blueprint.json'
    if output.exists() and any(output.iterdir()) and not marker.is_file():
        parser.error('Choose an empty output directory or a workspace created by this script')
    connection = json.loads(args.connection.read_text())
    response = Client(connection).call('catalog')
    if response.get('status') != 'success':
        raise RuntimeError(response.get('error', 'Cannot read the benchmark catalog'))
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix='qcalgym-blueprint-') as temp:
            source = args.source.resolve() if args.source else download_source(Path(temp))
            prepare_source(source, output)
    export(output, connection, response['experiments'])
    python = output / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not args.skip_install:
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(output / '.venv')
        subprocess.run([str(python), '-m', 'pip', 'install', '-c', str(output / 'qcalgym_constraints.txt'), '-e', str(output),
                        'langchain-openai==1.2.0', 'langchain-anthropic==1.4.1'], check=True)
        subprocess.run([str(python), str(output / 'run_qcalgym.py'), '--check'], check=True)
    print(f'Blueprint workspace: {output}')
    print(f'Tool definitions: {output / "qcalgym_tools.json"}')
    print(f'Launch: "{python}" "{output / "run_qcalgym.py"}"')
    if args.run:
        command = [str(python), str(output / 'run_qcalgym.py')]
        if args.model:
            command += ['--model', args.model]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
