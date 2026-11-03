# On-Call Rotation

Owner: Platform Engineering. Every engineer on a product team that runs production services joins the on-call rotation after completing onboarding.

## Schedule

Shifts last one week and hand over every Monday at 10:00 local time. Each shift has a primary and a secondary responder. The secondary is paged if the primary does not acknowledge an alert within 15 minutes. Rotations are managed in PagerDuty; swap shifts there and tell your team in Slack.

## Expectations

While on call as primary you should be able to reach a laptop within 15 minutes of a page. Acknowledge pages promptly, even if you are still investigating. If an alert indicates a security problem, follow the security incident response runbook and escalate to the security on-call.

Escalate to your engineering manager if an incident is customer-facing and lasts longer than 30 minutes, or if you need help from another team.

## Compensation for on-call

Engineers receive an on-call stipend of 400 USD for each full week as primary and 200 USD as secondary. If you are paged outside working hours, you may take an equivalent amount of time off in lieu during the following two weeks.

## Handover

At the end of each shift, write a short handover note in the #oncall-handover channel. List open incidents, noisy alerts that should be tuned, and any follow-up tickets you created.

## Alert hygiene

An alert that fires more than three times in a week without requiring action must be fixed or deleted. Review alert noise in the monthly reliability review.
