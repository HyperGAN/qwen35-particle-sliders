"""Bounded search with source/data-bound reuse and a common OFF comparison."""
import json
import math
from statistics import mean
from pathlib import Path
import subprocess
import sys

from .data import digest, write_json
from .math_evaluation import evaluate_math
from .training import implementation_identity, train_particle
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
    if (contract.get('upstream') != upstream or result.get('upstream') != upstream
            or contract.get('implementation') != implementation_identity()
            or contract.get('data_sha256') != digest(config['data'])):
        raise ValueError('Cached source/data identity differs; use a fresh output directory')
    if (result.get('selected') != 'raw' or result.get('metric') != 'raw_hidden_rms'
            or not contract.get('development_sha256')):
        raise ValueError('Cached result lacks raw held-out metric identity')
    if result.get('steps') != config['steps'] or result.get('completed_steps') != config['steps']:
        raise ValueError('Run did not complete the declared training horizon')
    if result.get('raw_sha256') != digest(root/'particle.safetensors'):
        raise ValueError('Cached RAW artifact differs from the completed run')
    off = result['initial']
    for key in ('raw', 'student'):
        values = result[key]
        if any(values[field] != off[field] for field in
               ('off_rms', 'off_sse', 'compared_values', 'rows')):
            raise ValueError('ON/OFF baselines or response dimensions differ')
        for name in ('on_rms', 'off_rms', 'on_sse', 'off_sse'):
            if not math.isfinite(values[name]) or values[name] < 0:
                raise ValueError('Held-out errors must be finite and nonnegative')
    comparison = {key: contract[key] for key in
                  ('identity', 'data_sha256', 'upstream', 'implementation', 'development_sha256', 'names', 'rows')}
    comparison['off'] = {key: off[key] for key in ('off_rms', 'off_sse', 'compared_values', 'rows')}
    return result, comparison


def finished_answers(workspace, path, development, *, device, teacher=None):
    strength = 1.0 if teacher else 0.0
    if not path.exists():
        evaluate_math(workspace, path, device=device, teacher=teacher, strength=strength, **development)
    report = json.loads(path.read_text())
    artifact = report['artifact']
    protocol = report['protocol']
    expected = digest(teacher) if teacher else None
    if (artifact['teacher_sha256'] != expected or artifact['strength'] != strength
            or artifact['deltas_sha256'] is not None or artifact['norm_gain'] != 1.0
            or protocol['seed'] != development['seed']
            or protocol['selection_offset'] != development['offset']
            or protocol['generation']['max_tokens'] != development['max_tokens']
            or len(protocol['indices']) != development['limit']):
        raise ValueError('Saved finished answers differ from the declared artifact or evaluation')
    generations = path.with_suffix('.generations.jsonl')
    if report.get('generations_sha256') != digest(generations):
        raise ValueError('Saved finished-answer file differs from its evaluation report')
    rows = [json.loads(line) for line in generations.read_text().splitlines()]
    if ([row['example_index'] for row in rows] != protocol['indices']
            or len(set(protocol['indices'])) != len(rows)
            or any(row['sample'] != 0 or not 0 <= row['score'] <= 1 for row in rows)):
        raise ValueError('Finished-answer rows differ from the declared matched cases')
    identity = {key: report[key] for key in
                ('model_id', 'source_version_id', 'config_id', 'validation_sha256', 'runtime', 'protocol')}
    scores = None if any(row['truncated'] for row in rows) else [row['score'] for row in rows]
    return scores, identity


def search(spec, *, guard_for=None, finished_probe=None):
    from particle_sliders.candidate_failure import CandidateFailure, CandidateGuard, FileRecord, GuardUnmeasured

    guarded = guard_for is not None or finished_probe is not None
    if guarded and (not callable(guard_for) or not callable(finished_probe)):
        raise GuardUnmeasured("Qwen3.5 search requires a text candidate guard and finished-output probe")
    if not guarded and spec['base'].get('device') != 'cpu':
        raise GuardUnmeasured("Qwen3.5 search has no bound qualified finished-text probe")
    if not 1 <= len(spec['trials']) <= 12 or not 1 <= spec['final_steps'] <= 1600:
        raise ValueError('Declared local search cap exceeded')
    development = spec['development']
    if (set(development) != {'offset', 'limit', 'seed', 'max_tokens'}
            or any(type(value) is not int for value in development.values())
            or not 0 <= development['offset'] < development['offset'] + development['limit'] <= 16
            or development['seed'] != 1000 or development['max_tokens'] != 8192):
        raise ValueError('Use the predeclared development slice and generation protocol')
    root = Path(spec['output'])
    root.mkdir(parents=True, exist_ok=True)
    upstream = load(spec['base'].get('upstream')).identity
    records, reference = [], None
    base = dict(spec['base'])
    from particle_sliders.weight_policy import require_no_weight_ema
    require_no_weight_ema(base.get('ema_decay'))
    base['ema_decay'] = None
    configs = [dict(base, **trial, output=str(root/f'trial-{index:02d}'))
               for index, trial in enumerate(spec['trials'])]
    for config in configs:
        require_no_weight_ema(config.get('ema_decay'))
        if not 1 <= config['steps'] <= 256:
            raise ValueError('Search trial must be bounded to 1..256 steps')
        if (Path(config['output'])/'result.json').exists():
            checked_result(config, upstream)

    def run(config, path, log_path):
        guard = guard_for(config) if guarded else None
        if guarded and (not isinstance(guard, CandidateGuard) or guard.binding['media'] != 'text'):
            raise GuardUnmeasured("Qwen3.5 search requires a guard for each text candidate")
        if not (Path(config['output'])/'result.json').exists():
            write_json(path, config)
            if guard is None:
                with log_path.open('a') as log:
                    subprocess.run([sys.executable, '-m', 'qwen35_sliders.cli', 'train-particle', str(path)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                train_particle(config, guard=guard, finished_probe=finished_probe)
        result, comparison = checked_result(config, upstream)
        if guard is not None:
            if 'assessed_full' not in result:
                raise GuardUnmeasured("Qwen3.5 result has no assessed FULL checkpoint")
            checkpoint = FileRecord(**result['assessed_full'])
            expected = Path(config['output']) / f"assessed-full-{config['steps']:06d}.pt"
            if Path(checkpoint.path) != expected.resolve():
                raise GuardUnmeasured("Qwen3.5 result does not bind the final candidate checkpoint")
            checkpoint.verify()
            guard.require_selection(checkpoint)
        return result, comparison

    for index, config in enumerate(configs):
        try:
            result, comparison = run(config, root/f'trial-{index:02d}.json', root/f'trial-{index:02d}.log')
        except CandidateFailure as failure:
            records.append(dict(index=index, config=config, status='catastrophic_failure',
                                reason=str(failure), intended_effect=None))
            continue
        if reference is None:
            reference = comparison
        elif reference != comparison:
            raise ValueError('Trials do not share the same held-out data, base and OFF baseline')
        student = result['student']
        ratio = student['on_rms'] / student['off_rms'] if student['off_rms'] else None
        relative_sse = student['on_sse'] / student['off_sse'] if student['off_sse'] else None
        records.append(dict(index=index, config=config, result=result,
                            student_relative_rms=ratio, student_relative_sse=relative_sse))
        print(json.dumps(dict(trial=index, student_relative_rms=ratio)), flush=True)
    eligible = [record for record in records if record.get('status') != 'catastrophic_failure']
    if not eligible:
        write_json(root/'search-results.json', records)
        return
    workspace = base['workspace']
    device = base.get('device', 'cuda')
    off, output_reference = finished_answers(workspace, root/'development-off.json', development, device=device)
    for record in eligible:
        trial_root = Path(record['config']['output'])
        on, identity = finished_answers(workspace, trial_root/'development-on.json', development,
                                        device=device, teacher=trial_root/'particle.safetensors')
        if identity != output_reference:
            raise ValueError('Development evaluations differ in model, cases, generation or runtime')
        record['intended_effect'] = (None if on is None or off is None
                                     else mean(on_score - off_score for on_score, off_score in zip(on, off, strict=True)))

    write_json(root/'search-results.json', records)
    if any(record['intended_effect'] is None for record in eligible):
        return
    top = max(record['intended_effect'] for record in eligible)
    winners = [record for record in eligible if record['intended_effect'] == top]
    if len(winners) > 1:
        write_json(root/'selected.json', dict(trials=[record['index'] for record in winners],
                                             intended_effect=top))
        return

    best = winners[0]
    final = dict(best['config'], steps=spec['final_steps'], output=str(root/'final'))
    selected = dict(trial=best['index'], criterion='mean matched finished-answer score ON minus OFF',
                    intended_effect=top, config=final, comparison_identity=reference)
    final_path = root/'final.json'
    _, comparison = run(final, final_path, root/'final.log')
    if comparison != reference:
        raise ValueError('Final run does not share the trials\' held-out data, base and OFF baseline')
    write_json(root/'selected.json', selected)
