# Security Incident Response Runbook

Owner: Security Engineering. Use this runbook for any suspected security incident: leaked credentials, malware, unauthorized access or data exposure.

## Severity levels

- SEV1: confirmed breach of customer data or production systems. Page the security on-call immediately, any time of day.
- SEV2: suspected compromise that is contained, such as a single laptop with malware. Respond within one hour.
- SEV3: policy violations or low-risk findings. Respond within one business day.

## First steps

1. Do not power off an affected machine. Disconnect it from the network so memory can be captured for forensics.
2. Open an incident in the #sec-incidents Slack channel with the /incident command. This creates a ticket and a dedicated war room channel.
3. Name an incident commander. The incident commander coordinates the response and is the only person who communicates with leadership.
4. Rotate any credentials that may have been exposed. API keys found in a public repository must be revoked within 30 minutes of discovery.

## Containment and evidence

Preserve logs before making changes. Export the relevant audit logs from the SIEM and attach them to the incident ticket. Block malicious IP addresses at the edge firewall and disable compromised accounts in the identity provider.

## Notification

If customer personal data is involved, Legal must be told within 4 hours of the incident being declared, so that regulators can be notified within the 72-hour window required by GDPR. Never contact customers directly about an incident without approval from Legal and the incident commander.

## After the incident

Hold a blameless postmortem within five business days. The postmortem document lists the timeline, root cause, impact and follow-up actions, each with an owner and a due date.
