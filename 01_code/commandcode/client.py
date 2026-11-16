"""CommandCode adapter. Credentials: environment or current Windows user's DPAPI store."""
import json
import os
import subprocess
import urllib.request
import urllib.error

BASE = 'https://api.commandcode.ai/provider/v1'

def key():
    value = os.environ.get('COMMANDCODE_API_KEY')
    if value:
        return value
    script = r"""
$ErrorActionPreference='Stop'
$path=Join-Path $env:LOCALAPPDATA 'NIDSResearchSecrets/commandcode.dpapi'
$secret=Get-Content -LiteralPath $path | ConvertTo-SecureString
$ptr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try { [Console]::Write([Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
"""
    result = subprocess.run(['pwsh','-NoProfile','-Command',script], capture_output=True, text=True, encoding='utf-8')
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Credential missing: run configure_key.ps1')
    return result.stdout.strip()

def request(route, body=None):
    headers={'Content-Type':'application/json', 'User-Agent':'NIDS-Research-CommandCode/1.0'}
    if body is not None:
        headers['Authorization']='Bearer '+key()
    req=urllib.request.Request(BASE+route, headers=headers,
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Never print request headers or credentials, even on authentication errors.
        raw=exc.read().decode(errors='replace')
        secret=headers.get('Authorization','').removeprefix('Bearer ')
        if secret:
            raw=raw.replace(secret,'[REDACTED]')
        raise RuntimeError(f'HTTP {exc.code}: {raw[:1200]}') from None

def chat(model, messages):
    return request('/chat/completions', {'model':model, 'messages':messages,
        'max_completion_tokens':2048, 'reasoning_effort':'low'})
