# Engineering Glossary

Owner: Developer Experience. Short definitions of internal terms, services and error codes used across Northwind Labs.

## Services

- **Beacon**: the internal feature-flag service. Flags are evaluated client-side and cached for 60 seconds.
- **Granary**: the data warehouse that stores product analytics. Queries run on a nightly snapshot.
- **Ledgerly**: the expense and invoicing system used by Finance.
- **NorthGate**: the VPN client that manages WireGuard profiles.
- **Shipyard**: the deployment pipeline. Every merge to main is deployed to staging by Shipyard and promoted to production after canary checks pass.

## Terms

- **Canary**: a deployment to 5 percent of traffic that runs for 20 minutes before full rollout.
- **Code freeze**: the last two weeks of each quarter, when only approved hotfixes can merge.
- **Golden signals**: latency, traffic, errors and saturation, the four metrics every service dashboard shows.
- **Runbook**: a step-by-step guide for handling a specific alert or incident.

## Error codes

- **NW-1001**: request rejected because the tenant header is missing.
- **NW-2207**: the Granary snapshot is stale; the nightly export did not finish.
- **NW-3150**: a Beacon flag key was not found; the client fell back to the default value.
- **NW-4012**: the device certificate has expired; re-enroll the device.
- **NW-5003**: Shipyard rolled back a deployment because the canary error rate exceeded 2 percent.

When reporting a bug, always include the full NW error code and the request id from the response headers.
