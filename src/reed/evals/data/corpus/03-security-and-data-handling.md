# Security and Data Handling

Customers connect Windrose directly to their warehouses, so this policy is not
optional reading. It is owned by the Security Lead and reviewed quarterly;
privacy questions are owned by the Data Protection Officer.

## Data classification

Every piece of information at Meridian Labs sits in one of four tiers. If you
cannot tell which, treat it as one tier higher and ask the Security Lead.

**Tier 0 — Public.** The marketing site, published documentation, open-source
repositories, conference material. No restriction.

**Tier 1 — Internal.** Roadmaps, the org chart, meeting notes, internal metrics,
draft memos. Any employee may read Tier 1 data, and it may be shared freely
inside the company.

**Tier 2 — Client data.** Customer lineage graphs, uploaded schemas and table
metadata, query logs, support ticket attachments, and customer contact lists.
Tier 2 data must not be copied out of approved systems, and any local copy made
for debugging must be deleted within seven days. Which devices may open Tier 2
data is set by the remote work policy.

**Tier 3 — Restricted.** Production credentials, payroll records, signed customer
contracts, security incident files, and board material. Tier 3 access requires a
managed device, a hardware security key, and named approval from the Security
Lead, recorded in Strongbox. Approvals are time-boxed and never granted to a
shared account.

## Accounts and authentication

Every system is behind Compass single sign-on. Shared logins are prohibited
without exception.

Latchkey is mandatory for any credential that cannot live in Compass. Master
passphrases must be at least 14 characters. Hardware security keys (FIDO2) are
required for Compass, Strongbox, and the cloud console; every employee is issued
two keys, one of which should be kept outside your laptop bag.

Access is granted from a role template within one business day of your start
date, and revoked **within two hours of the end of your last working day**. The
Security Lead runs an access review every quarter and must complete it within 10
working days of quarter end.

## Devices

Company devices must run full-disk encryption, lock the screen after five minutes
of inactivity, and install operating-system security updates **within 14 days of
release**. For critical vulnerabilities — CVSS 9.0 or above — the window drops to
**seven days**, and Compass will force the update if you have not.

Report a lost or stolen device in the `#sec-report` channel immediately and to
the Security Lead **within one hour**, at any time of day. We will wipe it
remotely; a wiped device is a smaller problem than a delayed report.

## Handling client data

Production data is never copied into staging or development environments. Use the
anonymised fixture set; if it does not cover your case, extend the fixtures
rather than reaching for the real thing.

Client data is deleted **30 days after a contract ends**, and a deletion
certificate is issued to the customer within 45 days. Do not keep exports,
spreadsheets, or screenshots past those dates.

Sending Tier 2 data outside an approved system — email, a personal cloud drive, a
screenshot pasted into a public channel — is a reportable incident even if
nothing bad follows from it.

## Reporting a suspected breach

Any suspected exposure of Tier 2 or higher data must be reported to the Security
Lead immediately, without first checking whether it is real. Reporting something
that turns out to be nothing carries no consequence whatsoever; not reporting
does.

A confirmed exposure runs through the incident response process, which sets the
severity, the paging path, and the customer communication. Where personal data is
involved, the **Data Protection Officer files a notification with the supervisory
authority within 72 hours** of the company becoming aware of it. That clock is
legal, not internal, and it starts at first awareness rather than at
confirmation.

## Phishing and social engineering

Forward anything suspicious to `#sec-report` and delete nothing until the
Security Lead has looked at it. We run a simulated phishing exercise every
quarter. Clicking a simulation earns a 20-minute refresher session and nothing
else — there is no disciplinary consequence, and no list of names is published.

Nobody at Meridian Labs will ever ask you for a password, a security key touch,
or a one-time code over chat, phone, or email. A request that claims to come from
a founder and is urgent is the attack we see most often.

## Training and vendors

Security onboarding is completed in your first week, and an annual refresher is
due within 30 days of your start-date anniversary.

Any tool that will touch Tier 2 or higher data requires a security review by the
Security Lead and a signed data processing agreement **before** purchase,
regardless of what it costs. Buying first and asking afterwards is the one
procurement mistake we treat as serious.
