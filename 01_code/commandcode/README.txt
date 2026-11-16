CommandCode local connection

API base: https://api.commandcode.ai/provider/v1
Default smoke model: gpt-6-luna (as advertised by the provider)
No external Python dependencies; Python 3.12 standard library.

Run from <WORKSPACE>:
  py -3.12 tools/commandcode/smoke.py

This sends one connectivity prompt followed by the first three fixed D4 cohort A
requests (30% missing packets). It does not launch the full 1000-flow experiment.
Every execution incurs up to four model calls. HTTP errors stop the run.
Reports are written to outputs/d4_teacher_api_20261114/commandcode_smoke/.
The existing A sample hash and common prompt are reused. Missing values and labels
are not provided to the API. Small-sample metrics are connectivity diagnostics,
not a teacher ranking or downstream classifier evaluation.

Credentials are read from COMMANDCODE_API_KEY, or from Windows DPAPI encrypted
storage outside the repository under LOCALAPPDATA/NIDSResearchSecrets.
To replace a key, run configure_key.ps1 with pwsh and enter it at the hidden prompt.
The key previously provided in chat should be rotated in CommandCode Studio.
Do not commit credentials or transfer the encrypted store to another machine.

The advertised and returned model IDs are recorded. This is a third-party
provider connection and does not independently certify the upstream model.
