# Data Retention Policy

Owner: Legal and Security Engineering. This policy sets how long Northwind Labs keeps each category of data and how it is destroyed.

## Retention periods

| Data category | Retention period |
| --- | --- |
| Customer account data | Life of the contract plus 90 days |
| Application logs | 30 days |
| Security audit logs | 1 year |
| Database backups | 35 days |
| Financial records and invoices | 7 years |
| Employee personnel files | 6 years after employment ends |
| Recorded sales calls | 12 months |

## Customer deletion requests

When a customer closes their account, their data is deleted from production systems within 30 days and ages out of backups within a further 35 days. A data subject request under GDPR or CCPA must be completed within 30 days; route it to the privacy team through the Legal intake form.

## Legal hold

A legal hold overrides every retention period. When Legal places a hold, affected data must not be deleted or modified until Legal releases the hold in writing, even if its retention period has expired.

## Destruction

Data is deleted using the platform's secure deletion tooling. Laptops and storage devices are wiped to NIST 800-88 standards before reuse and physically shredded at end of life by a certified vendor.

## Exceptions

Requests to keep data longer than this policy allows must be approved by the Data Protection Officer and reviewed every year.
