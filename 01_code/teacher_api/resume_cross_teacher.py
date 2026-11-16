"""Resume the frozen batch; probe the prior 402 slot before new requests."""
import cross_teacher_chain as c

original_run_phase = c.run_phase


def resume_phase(phase, model, rows, bodies):
    if phase == 'A':
        prior = [c.read(p) for p in sorted((c.DEST/'A/plain').glob('[0-9][0-9][0-9][0-9].json'))]
        c.g.save(c.DEST/'resume_20261115.json', {
            'utc': c.utc(), 'authorization': 'User requested API resume after suspected balance exhaustion',
            'prior_records': len(prior), 'prior_http_success': sum(r['http_status'] == 200 for r in prior),
            'unchanged': 'Frozen samples, masks, prompts, model, output cap, scoring and B selection',
            'action': 'Retry prior 402 once via existing bounded recovery before requesting missing slots; retain all HTTP-200 answers',
        })
        for r in prior:
            if r['http_status'] == 402 and not r.get('recovery_performed'):
                c.state('A_balance_recovery_probe', draw=r['draw'], model=model)
                c.one(model, 'A', 'plain', rows[r['draw']], bodies['plain'][r['draw']], recovery=True)
                if c.STOP.is_set():
                    raise RuntimeError('API still rejects recovery; no new slots launched')
    return original_run_phase(phase, model, rows, bodies)


if __name__ == '__main__':
    c.run_phase = resume_phase
    c.main()
