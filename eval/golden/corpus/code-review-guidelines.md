# Code Review Guidelines

Owner: Engineering Council. These guidelines apply to every repository in the Northwind GitHub organization.

## Approvals

Every pull request needs at least one approval from a code owner before merging. Changes to authentication, billing or data deletion code need two approvals, one of them from a member of Security Engineering. Authors may not approve their own pull requests, and nobody may bypass branch protection on the main branch.

## Size

Keep pull requests small: aim for under 400 changed lines. Large refactors should be split into a series of reviewable steps. Generated files and lockfiles do not count towards the limit but must be called out in the description.

## Turnaround

Reviewers should give a first response within one business day. If you cannot review in time, say so and suggest another reviewer. Authors should respond to every comment, either with a change or with a reason for not making one.

## What reviewers check

- Correctness, including edge cases and error handling.
- Tests: new behaviour needs tests, and bug fixes need a regression test.
- Security: no secrets in code, input validation at trust boundaries, and parameterized queries.
- Readability and naming. Prefer clear code over clever code.
- Observability: new services emit structured logs and metrics.

## Comment etiquette

Prefix optional suggestions with "nit:" so the author knows they are not blocking. Critique the code, never the person. Approve once the remaining comments are only nits.

## Continuous integration

CI must be green before merging. Flaky tests are quarantined within one day and fixed within a week; never retry a failing pipeline repeatedly to get a pass. Merges to main are blocked during the end-of-quarter code freeze except for approved hotfixes.
