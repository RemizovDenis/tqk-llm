# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- New `tqk benchmark` CLI command for local roundtrip/latency/size audits.
- JSON + Markdown benchmark artifacts (`tqk_benchmark.json`, `tqk_benchmark.md`).

### Tests
- Added benchmark module and CLI tests.

## [0.1.1] - 2026-04-04

### Changed
- Switched package identity to `tqk-llm` to avoid collision with unrelated `tqk` package on PyPI.
- Reworked README installation path to source-based setup with explicit package-name note.
- Added GitHub project standards: CODEOWNERS, PR template, issue templates, support, and code of conduct.

### CI/Quality
- Strict mypy compatibility fix for Python 3.11 CI.
- Repository formatting normalized with Ruff.

## [0.1.0] - 2026-04-01

### Added
- Initial public release of TQK format and validation pipeline.
