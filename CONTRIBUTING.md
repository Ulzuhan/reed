# Contributing

Reed is a single-author project, developed in the open as a Hesperia Labs case study. Issues and
contributions are welcome all the same.

## Issues

Bug reports, questions and feature suggestions all go through
[GitHub issues](https://github.com/Ulzuhan/reed/issues). For a suspected security vulnerability,
use the private reporting flow in [SECURITY.md](SECURITY.md) instead of a public issue.

## Pull requests

Open an issue first for anything larger than a typo fix, so the approach is agreed before the
work exists. Every pull request runs the full quality gate — lint, types, tests on Python
3.11–3.13, browser E2E, dependency and secret scans, a Docker smoke test and CodeQL — and merges
only when all of it is green.

Local setup mirrors CI:

```bash
make install   # uv sync, including dev tools
make lint type # ruff + mypy, same configuration as CI
make test      # full pytest suite
```

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
