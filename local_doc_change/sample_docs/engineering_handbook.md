# Engineering Handbook Overview

This handbook describes how the Nimbus Robotics engineering organization builds, ships, and operates software. It applies to all full-time engineers, contractors, and interns who contribute to production systems. The handbook is reviewed every quarter by the engineering leadership team, and amendments are announced in the #engineering channel. When this document conflicts with a team-specific runbook, the team runbook takes precedence for that team only.

# Source Control and Branching Strategy

We use trunk-based development with short-lived feature branches. Feature branches should be merged into main within three days to avoid large, hard-to-review changes. Branch names follow the pattern `type/short-description`, for example `feat/gripper-calibration` or `fix/telemetry-drift`. Long-running branches require team lead approval and a documented integration plan. Direct commits to main are prohibited; all changes flow through a pull request. Rebasing onto the latest main before merge is encouraged to keep history linear.

# Code Review Policy

Every pull request must receive at least one approval from a senior engineer before it can be merged. Reviewers are expected to respond to review requests within one business day. Reviews should focus on correctness, readability, test coverage, and operational risk rather than personal style preferences. Authors must keep pull requests under 400 lines of change where possible, splitting larger work into reviewable increments. A pull request that touches authentication, billing, or robot-motion code additionally requires review from a domain owner. Draft pull requests do not require review until marked ready.

# Testing Standards

All new behavior must be covered by automated tests at the appropriate level: unit tests for pure logic, integration tests for service boundaries, and end-to-end tests for critical user journeys. The continuous integration pipeline must pass before a pull request can be merged. Flaky tests are treated as production incidents and are quarantined within one business day of detection. Code coverage should not decrease as a result of a change, and motion-control modules must maintain at least eighty percent line coverage.

# Continuous Integration and Delivery

Every push runs linting, type checks, unit tests, and a security scan. A green build is required before merge. Merging to main triggers an automatic deployment to the staging environment, where smoke tests run against simulated hardware. Promotion from staging to production is a manual, audited step performed through the deploy dashboard. Build artifacts are immutable and tagged with the commit hash so any release can be traced back to its exact source.

# Deployment Freeze

Production deployments are frozen every Friday after 2 PM to avoid weekend incidents when on-call coverage is thinner. Emergency hotfixes require director approval during a freeze and must be accompanied by a rollback plan. The freeze also applies to the days before a major company holiday. Routine configuration changes that do not alter behavior are exempt but must still be logged.

# Release Cadence

Nimbus ships a minor release every two weeks, aligned with the sprint boundary. Major releases are planned quarterly and announced to customers four weeks in advance. Every release has a designated release captain who owns the changelog, the go/no-go decision, and post-release monitoring. Customer-facing release notes are written in plain language and reviewed by the product team before publication.

# On-Call Rotation

The on-call engineer must acknowledge all production pages within 15 minutes during business hours. The rotation runs on a weekly basis, with each engineer taking a full week of primary duty followed by a week of secondary backup. The rotation schedule is published two weeks in advance, and swaps must be arranged directly between engineers and recorded in the scheduling tool. After any week of primary on-call, the engineer is entitled to a recovery day at their discretion.

# Incident Management

Incidents are classified as SEV1 for a full outage, SEV2 for major degradation, and SEV3 for a minor issue with a workaround. SEV1 incidents require an immediate page to the entire engineering leadership team and the opening of a dedicated incident channel. A blameless post-mortem is required for all SEV1 and SEV2 incidents within 48 hours of resolution. Post-mortems document the timeline, contributing factors, and concrete follow-up actions with owners and due dates.

# Observability and Monitoring

Every production service must emit structured logs, metrics, and traces. Each service owns a dashboard showing its key health indicators and a documented set of alerts tied to user-facing symptoms rather than internal causes. Alerts must be actionable; any alert that fires without a clear response is reviewed and either fixed or removed. Service level objectives are defined per service and reviewed monthly against actual performance.

# Technical Documentation

Every service has a README describing its purpose, owners, runbook, and local setup instructions. Architecture decisions are recorded as dated decision records stored alongside the code. Documentation is treated as part of the definition of done; a feature is not complete until its operational documentation is updated. Stale documentation should be flagged or fixed whenever it is encountered.

# Security Practices in Engineering

Secrets are never committed to source control and are managed through the central secrets manager. Dependencies are scanned for known vulnerabilities on every build, and critical advisories are patched within seventy-two hours. All external input is validated at trust boundaries, and database access uses parameterized queries exclusively. Engineers complete secure-coding training during onboarding and annually thereafter.
