# Security policy

## Supported versions

Security fixes are applied to the current `main` branch and the latest published container image.
Reed is pre-1.0, so operators should track the latest release rather than expect maintenance of
older minor versions.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's **Report a
vulnerability** form in the repository's Security tab to start a private security advisory. Include
the affected version or commit, deployment profile, reproduction steps, impact, and any proposed
mitigation. You should receive an acknowledgement within seven days.

Do not include real API keys, private documents, personal data, or production endpoints in a
report. Use synthetic data and revoke any credential that may have been exposed.

## Deployment boundary

Reed processes untrusted document contents, but it is not a multi-tenant authorization system.
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
