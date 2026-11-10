# Evidence archives

These ZIPs preserve exact original bytes while keeping manual GitHub uploads manageable.
- first_round_predictions_and_masks.zip: 132-cell metrics, per-flow predictions, shared loss masks and 12 imputed arrays.
- revised_predictions.zip: 66-cell evaluation of the three corrected classifiers.
- classifier_checkpoints.zip: six first-round + three corrected classifier checkpoints.
- execution_code.zip: actual evaluation / classifier code snapshots and retained logs, plus a current copy of the DSH training source. The latter is not proof of startup-time source freezing.
- early_helpers_and_reference_baselines.zip: the former third/fifth upload batches, archived as reference rather than labelled Nuwa author source.

The original protocol train/eval hashes differ from repaired source; both are preserved, and this deviation is disclosed in the audit.
Four full trained MLM checkpoints remain on the experiment server and are not included here. Incomplete author-weight downloads are not packaged as usable models.
Full retraining is not claimed to work from this upload package alone.
