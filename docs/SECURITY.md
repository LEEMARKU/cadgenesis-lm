# Security Policy

## Reporting a Vulnerability

Please do **not** open a public issue for security vulnerabilities. Report
them privately to the CADGenesis maintainers.

We ask that you:

- Provide a clear description of the vulnerability and its impact.
- Include steps to reproduce, affected versions, and any suggested mitigations.
- Allow time for triage and a fix before public disclosure.

## Scope

This policy covers the CADGenesis-LM source repository. Generated CAD models
and serialized TOON data are treated as untrusted input where noted in the
codebase; treat any code that parses external files (checkpoints, datasets,
TOON payloads) as a security boundary.

## Supported Versions

| Version   | Supported          |
| --------- | ------------------ |
| 6.0.x     | :white_check_mark: |
| 2.0.x     | :white_check_mark: |
| < 2.0     | :x:                |
