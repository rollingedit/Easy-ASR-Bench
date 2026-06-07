# Windows Release Matrix

These scripts collect evidence for release smoke rows that cannot be proven by unit tests alone.

Run from an expanded public release ZIP or repo checkout on the target Windows machine:

```powershell
powershell -ExecutionPolicy Bypass -File qa\windows_matrix\run_release_matrix.ps1 -Tag v0.3.4 -Output qa\windows_matrix\evidence
```

Every row should produce:

- command log
- environment summary
- result artifact or expected failure artifact
- SHA256 for each evidence file

Rows must stay `not_run` in `release-smoke-vX.Y.Z.json` until real evidence exists.
