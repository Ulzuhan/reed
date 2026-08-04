# Engineering Standards

These standards apply to every repository in the `meridian-labs` organisation.
They are owned by the Head of Engineering. Where a rule is enforced by Slipway,
our build and deployment pipeline, it is not a matter of taste — a red pipeline
blocks a merge, and there is no override.

## Languages and tooling

Backend services are Python 3.12; the Windrose front end is TypeScript. New
services in another language need a written case and the Head of Engineering's
agreement, because every language we add is a language somebody has to be on call
for at 03:00.

Formatting and linting run in Slipway on every push.

## Pull requests

Keep pull requests under **400 changed lines**. Above that, open it anyway but
explain in the description why it could not be split — generated code and
mechanical renames are perfectly good reasons.

**One approving review** is required to merge. **Two approving reviews** are
required for any change touching billing, authentication, or data export.
Authors never approve their own pull requests, and draft pull requests are exempt
until marked ready.

Reviewers give a first response **within one working day**. "First response" can
be "I cannot look at this until Thursday" — silence is the thing we are trying to
prevent.

## Testing

Changed files must reach **80% line coverage**; Slipway fails the build below
that threshold. Every bug fix ships with a regression test that fails without the
fix — if you cannot write one, say so in the pull request and explain why.

Integration tests run against ephemeral containers, never against a shared
environment. A test that is flaky twice in a week is quarantined by whoever
notices and fixed within 10 working days or deleted.

## Branching and merging

We work trunk-based. Branch from `main`, and merge back **within three working
days**. A weekly job deletes merged branches and any branch with no commits for
10 days.

Merges are squashed, and the commit subject carries the ticket identifier. The
body explains why, not what.

## Deploys

Every merge to `main` deploys to staging automatically. Production deploys are
manual, one click in Slipway, and permitted **07:00 to 17:00 WET, Monday to
Thursday**.

**No production deploys on Fridays, after 17:00, or at weekends**, with a single
exception: a fix for an active Sev-1, authorised by the Incident Commander in the
incident channel. The point is not that Friday code is worse. It is that Friday
evening is when nobody is left to notice.

Anything user-visible ships behind a feature flag. Flags are removed **within 60
days of full rollout**; Slipway posts a weekly list of flags older than 60 days,
and that list is expected to be empty.

## Production access

**Nobody holds standing production credentials.** Access is requested in
Strongbox for a window of at most **four hours**, requires a second engineer's
approval under the two-person rule, and every session is logged and reviewable.

Production credentials are Tier 3 data, so the conditions the security policy
attaches to that tier apply on top of the Strongbox request — check them before
you assume a laptop and a good reason are enough.

Read-only analytical questions should be answered against the anonymised replica
instead. Reaching for production because it is faster is how a routine Tuesday
becomes an incident.

## Data and migrations

Schema migrations are backwards compatible and two-phase: expand, deploy, migrate,
then contract in a later release. A destructive migration needs the Head of
Engineering's approval and a rollback that has been tested against a copy of
production-sized data.

Never copy production data into staging. The security policy is explicit about
this, and the anonymised fixture set exists so you do not have to.

## Dependencies

Update pull requests are raised weekly and automatically. Security patches for
vulnerabilities rated **CVSS 9.0 or above are merged within seven days**;
everything else within 30 days. A dependency that has been unmaintained for 12
months is a design decision, and it needs a note in the service README saying so.

## Shipping a new service

Before a new service takes production traffic it needs, at minimum: a runbook
page in Logbook, alert routes configured in Beacon, a dashboard someone has
actually looked at, and a named owning team. The Head of Engineering signs off
against that list.

Every service README documents how to run it locally, and that path must work for
a new joiner in under 30 minutes on a fresh laptop.

## Public API

The customer API is versioned in the URL path, currently `/v1/`. Breaking changes
require a **90-day deprecation notice** to customers, a written migration guide,
and deprecation headers on the affected endpoints for the whole notice period. We
have broken this rule once, in 2023, and we are still apologising for it.
