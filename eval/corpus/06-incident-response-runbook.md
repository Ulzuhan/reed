# Production Incident Response Runbook

Read this before your first on-call shift, not during it.

## On-call rotation

The rotation is one week long and hands over **Mondays at 10:00 WET (Lisbon
time)**. Two engineers are on it at any moment: the **Duty Engineer**, who takes
the page, and the **Backup Engineer**, who takes it when the primary does not.

To join the rotation you need **90 days of tenure** and **two completed shadow
rotations**, in which you receive every page and respond to none of them.

On-call is paid at **€300 per week**, rising to **€450** for a week containing a
public holiday in the engineer's country of residence. Night callouts also earn
time off in lieu — see the leave policy for how that is credited.

Swaps are arranged directly between engineers and recorded in Beacon at least 24
hours ahead. Nobody may hold the primary rotation for two consecutive weeks.

## Acknowledging a page

1. Acknowledge in Beacon **within five minutes**. Acknowledging is not fixing; it
   means a human has it.
2. An unacknowledged page escalates to the Backup Engineer after five minutes,
   and to the Head of Engineering ten minutes after that.
3. Open an incident channel in Wardroom named `#inc-<YYYYMMDD>-<short-name>` and
   post what you know, including "I do not yet know what this is".
4. Declare a severity. You can revise it later; do not delay it now.

Page through Beacon only, never individuals — the rotation exists so nobody has
to remember who is awake.

## Severity ladder

**Sev-1.** Windrose is unavailable or materially degraded for more than 5% of
customers, **or** any confirmed exposure of Tier 2 or higher data as defined in
the security policy. Response begins within 15 minutes, 24 hours a day. A
Signalpost entry goes up within 30 minutes and is updated every 30 minutes until
resolution.

**Sev-2.** A major feature is broken or degraded, with no data loss and a
workaround available. Response within one hour during business hours, taken as
07:00 to 20:00 WET, and the next business day otherwise. Signalpost is updated
within two hours.

**Sev-3.** Minor or cosmetic, workaround obvious. Handled the next business day.
No Signalpost entry.

When you cannot decide between two levels, take the higher one. Downgrading an
incident costs nothing; upgrading one at hour three costs a great deal.

## Roles during an incident

The **Incident Commander** owns the incident. By default this is the Duty
Engineer, but if you find yourself both commanding and typing fixes, hand command
to someone else immediately. The IC is the only person who declares the severity,
authorises a production deploy outside the normal deploy window, and declares the
incident resolved.

The **Comms Lead** writes Signalpost updates and answers the customer-facing
channels. The **Scribe** keeps a timestamped log in the incident channel. On a
Sev-1 all three roles must be different people.

Every decision goes in the incident channel. No incident discussion happens in
direct messages, ever.

## Data exposure path

Any incident involving Tier 2 or higher data pages the **Security Lead** in
parallel with the Duty Engineer and is a Sev-1 from the moment exposure is
confirmed.

The **Data Protection Officer** must be informed **within 60 minutes**. That is
not a formality: where personal data is involved, the regulatory notification
clock described in the security policy is already running.

Do not contact the affected customer yourself. Customer notification on a data
incident is written by the Comms Lead and approved by the DPO.

## Mitigation

Roll back first. **Any engineer may roll back a deploy without approval**, and
during a Sev-1 a rollback is always preferred to a forward fix, even when the fix
looks small. Feature flags are killed from Slipway and take effect within 60
seconds.

If the mitigation requires a production deploy outside the normal window, the
Incident Commander authorises it in the incident channel and it is exempt from
the usual deploy schedule.

## After it is over

The IC declares resolution in the incident channel and closes the Signalpost
entry.

Sev-1 and Sev-2 incidents require a written postmortem in Logbook **within five
working days**. Sev-1 postmortems get a 30-minute review meeting within 10
working days, open to the whole company.

Postmortems are blameless: they describe systems, alerts, and decisions made with
the information available at the time, and they do not name the individual who
ran the command. Every action item has a named owner and a due date, and **no
action item may go 30 days without a written update** on its Logbook page.

## What we measure

We target **99.9% monthly availability** for Windrose, which is an error budget
of roughly **43 minutes per month**. When a month's budget is spent, the next
sprint starts with reliability work before feature work.
