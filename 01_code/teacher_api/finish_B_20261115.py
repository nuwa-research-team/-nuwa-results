"""Observe the existing B queue, then finish its frozen recovery and scoring.

No successful HTTP-200 answer is reissued. No DeepSeek credentials are needed.
This supervisor does not modify the frozen experiment or report scripts.
"""
import concurrent.futures as cf
import ctypes
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
import cross_teacher_chain as c

OLD_PID = 47368
OUT = c.DEST / 'completion_20261115'
OUT.mkdir(exist_ok=True)


def save(name, value):
    c.g.save(OUT / name, value)


def snapshot(stage):
    variants = {}
    for variant in ('plain', 'behavior'):
        records = [c.read(p) for p in (c.DEST/'B'/variant).glob('[0-9][0-9][0-9][0-9].json')]
        variants[variant] = {
            'completed': len(records),
            'valid': sum(r['values'] is not None for r in records),
            'errors': dict(Counter(r['validation_error'] for r in records if r['validation_error'])),
        }
    save('progress.json', {'utc': c.utc(), 'stage': stage, 'variants': variants})


def wait_original():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, OLD_PID)
    if not handle:
        if ctypes.get_last_error() != 87:
            raise ctypes.WinError(ctypes.get_last_error())
        return
    try:
        while kernel.WaitForSingleObject(handle, 30000) == 258:
            snapshot('observing_original_queue')
    finally:
        kernel.CloseHandle(handle)


def finish():
    save('amendment.json', {
        'utc': c.utc(),
        'reason': 'Original coordinator is waiting in ThreadPoolExecutor shutdown; workers continue but status is stale.',
        'policy': 'Wait for original PID exit; verify frozen inputs; fill missing records and perform only original bounded transport recovery; retain all HTTP-200 answers; use original scorer. For a missing record, original attempt count is unknown: reserve its original three attempts and allow only the remaining three-attempt recovery.',
        'unchanged': 'teacher, samples, masks, prompts, output cap, four workers, two-second pacing, classifiers, invalid-output scoring',
    })
    wait_original()
    snapshot('original_process_exited')
    if (c.DEST/'B'/'comparison.json').exists():
        save('complete.json', {'utc': c.utc(), 'source': 'original_runner'})
        return
    _, _, rows, bodies = c.prepare()
    model = c.read(c.DEST/'selection.json')['winner']
    assert model != c.MODEL, 'This continuation is ADC-only.'
    lock = c.DEST/'running.lock'
    if lock.exists():
        assert int(lock.read_text()) == OLD_PID, 'Another owner holds the lock.'
        lock.unlink()
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    exceptions = []
    try:
        for recovery in (False, True):
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {}
                for i, row in enumerate(rows):
                    variants = list(bodies)
                    if i % 2:
                        variants.reverse()
                    for variant in variants:
                        path = c.DEST/'B'/variant/f"{row['draw']:04d}.json"
                        is_recovery = recovery or not path.exists()
                        futures[pool.submit(c.one, model, 'B', variant, row, bodies[variant][i], is_recovery)] = (i, variant)
                pending = set(futures)
                while pending:
                    done, pending = cf.wait(pending, timeout=30)
                    for future in done:
                        try:
                            future.result()
                        except Exception as exc:
                            i, variant = futures[future]
                            exceptions.append({'draw': i, 'variant': variant, 'recovery_pass': recovery,
                                               'error_type': type(exc).__name__, 'utc': c.utc()})
                            save('worker_exceptions.json', exceptions)
                    snapshot('transport_recovery' if recovery else 'fill_missing_records')
            if c.STOP.is_set():
                raise RuntimeError('Fatal API response; inspect sanitized records')
        for variant in bodies:
            for row in rows:
                assert (c.DEST/'B'/variant/f"{row['draw']:04d}.json").exists(), 'Missing response slot'
        c.g.save(c.DEST/'B'/'api_complete.json', {'utc': c.utc(), 'model': model,
                                               'completion_owner': 'finish_B_20261115'})
        c.state('B_scoring', model=model)
        subprocess.run([str(c.CPU), str(c.ROOT/'report_cross_teacher.py'), 'B'], check=True)
        subprocess.run([str(c.CPU), str(c.ROOT/'summarize_B_20261115.py')], check=True,
                       stdout=(OUT/'metric_audit.log').open('w', encoding='utf-8'))
        c.state('complete', winner=model)
        save('complete.json', {'utc': c.utc(), 'source': 'continuation', 'worker_exceptions': len(exceptions)})
    except Exception as exc:
        save('failure.json', {'utc': c.utc(), 'error_type': type(exc).__name__})
        raise
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    finish()
