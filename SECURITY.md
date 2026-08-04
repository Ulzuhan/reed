# Security policy

## Supported versions

Security fixes are applied to the current `main` branch and the latest published container image.
Reed is pre-1.0, so operators should track the latest release rather than expect maintenance of
older minor versions.

Reed v0.3 is the supported release line. The v0.1.x and v0.2.x tag contents remain MIT-licensed,
but are not maintained security branches.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's **Report a
vulnerability** form in the repository's Security tab to start a private security advisory. Include
the affected version or commit, deployment profile, reproduction steps, impact, and any proposed
mitigation. You should receive an acknowledgement within seven days.

Do not include real API keys, private documents, personal data, or production endpoints in a
report. Use synthetic data and revoke any credential that may have been exposed.

## Deployment boundary

Reed processes untrusted document contents, but it is not a multi-tenant authorization system or
a distributed service. Run one Reed API process per registry/active index; multiple workers can
race the SQLite-backed durable queue and are outside the supported security boundary.
One configured `REED_API_KEY` protects the whole `/v1` API, and anyone holding it can list, query,
upload, and delete every document in that instance. Put internet-facing deployments behind TLS and
an authenticated reverse proxy, keep the service bound to loopback by default, and add shared rate
limiting at the proxy when running multiple processes.

The PDF/text parser runs in a separate process with time, CPU, memory, and file-descriptor limits.
That reduces parser blast radius; it is not a substitute for an OS sandbox in hostile multi-tenant
environments. For that threat model, run the service in its hardened container (read-only root,
dropped capabilities, no-new-privileges), isolate its network, and use a dedicated low-privilege
host or VM.

Document excerpts are untrusted prompt data. Reed keeps them out of the system message and audits
citation structure, cited figures, and verbatim quotes. Model output can still be wrong: citations
are evidence links, not a cryptographic proof of semantic entailment.

Backups contain the original documents, registry metadata and embedded vectors. Create them only
while Reed is stopped, protect them like the source corpus, verify their checksums before restore,
and remember that remote Qdrant requires a separate coordinated snapshot. See the
[operations runbook](docs/runbooks.md).

## Repository protections

Release tags cannot be deleted or rewritten, the release environment deploys only from a `v*` tag,
and packaging runs behind the same quality gate as every pull request. Third-party actions and
container inputs are pinned by digest, dependencies and built images are scanned, and published
artifacts carry build provenance attestations you can check with
`gh attestation verify <file> --repo Ulzuhan/reed`.

Secret scanning and push protection are enabled, which covers GitHub's partner patterns. Detection
of non-provider patterns and validation of leaked credentials belong to GitHub Secret Protection,
which this repository does not have. Every pull request scans the whole repository history with
gitleaks, whose rules are not limited to partner providers, but that gate runs before a merge
rather than at push time. Keep `REED_API_KEY` and any provider credential out of commits, and
rotate anything that reaches the history.
