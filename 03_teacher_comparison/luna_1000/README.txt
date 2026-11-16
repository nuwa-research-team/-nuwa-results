D4 teacher comparison — CommandCode Luna / Astra

Current scope: user selected Luna only after Astra returned HTTP 403 MODEL_NOT_IN_PLAN.
Astra is not running. No automatic plan upgrade or billing change was made.

Frozen comparison: original A cohort, 1000 unique D4 flows, 30% packet masking,
10488 identical missing positions, identical system/user prompts to earlier Gemini/DeepSeek.
No retraining. Only masked inputs are sent; true hidden lengths, class labels, and flow IDs are not sent.

Luna: up to six concurrent API calls; 4096 completion-token cap; low reasoning;
strict JSON validation. Invalid completed answers are retained and zero-filled in scoring,
not repaired or repeatedly queried. Existing successful responses are reused on resume.

progress.json: current counts, measured throughput and estimated API time remaining.
gpt-6-luna/: per-flow responses and request hashes, attempt journals, final scores.json.
comparison.html: three-line summary and per-classifier table; refreshed automatically at completion.
summary.csv / classifiers.csv / comparison.json: exportable final comparison.
active_models_complete.json: written only after the selected Luna run AND scoring complete.
complete.json: reserved for both models completing; will not be written for Luna-only scope.

Mean accuracy and mean macro-F1 use equal weights across six frozen classifiers.
Magnitude MAE excludes sign; signed MAE includes sign. All figures use the same 1000 flows.
Partial Luna results are not ranked against complete 1000-flow results.
Historical test-pool model selection is exploratory, not an independent final evaluation.
CommandCode model names are provider-reported; upstream identity is not independently attested.

The API credential stays in the current user's encrypted DPAPI store, outside the repository.
The run uses local CPU only for classifier evaluation; no rented GPU is required.
