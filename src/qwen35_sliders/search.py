"""Bounded search with source/data-bound reuse and a common OFF comparison."""
import json
import math
from pathlib import Path
import subprocess
import sys

from .data import digest, write_json
from .training import implementation_identity
from .upstream import load


def checked_result(config, upstream):
    root = Path(config['output'])
    contract_path, result_path = root/'contract.json', root/'result.json'
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    if contract['config'] != config:
        raise ValueError('Cached recipe differs; use a fresh output directory')
    if result.get('contract_sha256') != digest(contract_path):
        raise ValueError('Result is not bound to its saved run contract')
    if (not contract.get('training_sha256') or not contract.get('prepared_pairs_sha256')
            or contract['prepared_pairs_sha256'] != digest(root/'prepared-pairs.json')):
        raise ValueError('Cached paired training state manifest differs or is missing')
    if (contract.get('upstream') != upstream or result.get('upstream') != upstream
            or contract.get('implementation') != implementation_identity()
            or contract.get('data_sha256') != digest(config['data'])):
        raise ValueError('Cached source/data identity differs; use a fresh output directory')
    if result.get('metric') != 'raw_hidden_rms' or not contract.get('development_sha256'):
        raise ValueError('Cached result lacks raw held-out metric identity')
    if result.get('steps') != config['steps'] or result.get('completed_steps') != config['steps']:
        raise ValueError('Run did not complete the declared training horizon')
    off = result['initial']
    for key in ('raw', 'ema', 'student'):
        values = result[key]
        if any(values[field] != off[field] for field in
               ('off_rms', 'off_sse', 'compared_values', 'rows')):
            raise ValueError('ON/OFF baselines or response dimensions differ')
        for name in ('on_rms', 'off_rms', 'on_sse', 'off_sse'):
            if not math.isfinite(values[name]) or values[name] < 0:
                raise ValueError('Held-out errors must be finite and nonnegative')
        if values['off_rms'] == 0 or values['off_sse'] == 0:
            raise ValueError('Zero OFF error cannot define a relative search metric')
    comparison = {key: contract[key] for key in
                  ('identity', 'data_sha256', 'upstream', 'implementation', 'training_sha256',
                   'development_sha256', 'prepared_pairs_sha256', 'names', 'rows')}
    comparison['off'] = {key: off[key] for key in ('off_rms', 'off_sse', 'compared_values', 'rows')}
    return result, comparison


def search(spec):
    if not 1 <= len(spec['trials']) <= 12 or not 1 <= spec['final_steps'] <= 1600:
        raise ValueError('Declared local search cap exceeded')
    root = Path(spec['output'])
    root.mkdir(parents=True, exist_ok=True)
    upstream = load(spec['base'].get('upstream')).identity
    records, reference = [], None
    configs = [dict(spec['base'], **trial, output=str(root/f'trial-{index:02d}'))
               for index, trial in enumerate(spec['trials'])]
    for config in configs:
        if not 1 <= config['steps'] <= 256:
            raise ValueError('Search trial must be bounded to 1..256 steps')
        if (Path(config['output'])/'result.json').exists():
            checked_result(config, upstream)

    def run(config, path, log_path):
        if not (Path(config['output'])/'result.json').exists():
            write_json(path, config)
            with log_path.open('a') as log:
                subprocess.run([sys.executable, '-m', 'qwen35_sliders.cli', 'train-particle', str(path)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return checked_result(config, upstream)

    for index, config in enumerate(configs):
        result, comparison = run(config, root/f'trial-{index:02d}.json', root/f'trial-{index:02d}.log')
        if reference is None:
            reference = comparison
        elif reference != comparison:
            raise ValueError('Trials do not share the same held-out data, base and OFF baseline')
        ratio = result['student']['on_rms']/result['student']['off_rms']
        records.append(dict(index=index, config=config, result=result,
            student_relative_rms=ratio, student_relative_sse=result['student']['on_sse']/result['student']['off_sse']))
        print(json.dumps(dict(trial=index, student_relative_rms=ratio)), flush=True)
    best = min(records, key=lambda row:row['result']['student']['on_sse'])
    final = dict(best['config'], steps=spec['final_steps'], output=str(root/'final'))
    # A fresh final starts at the common seed, with the selected setting's horizon.
    selected = dict(trial=best['index'], criterion='lowest raw held-out student SSE on identical rows/OFF baseline',
                    config=final, comparison_identity=reference)
    final_path = root/'final.json'
    if (Path(final['output'])/'result.json').exists():
        _, comparison = checked_result(final, upstream)
    else:
        write_json(root/'selected.json', selected)
        _, comparison = run(final, final_path, root/'final.log')
    if comparison != reference:
        raise ValueError('Final run does not share the trials\' held-out data, base and OFF baseline')
    write_json(root/'search-results.json', records)
    write_json(root/'selected.json', selected)
