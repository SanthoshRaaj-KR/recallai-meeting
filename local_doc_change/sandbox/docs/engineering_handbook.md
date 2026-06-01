# Deployment Process

Production deployments are performed via the CI/CD pipeline. All deployments require a green build and must be triggered through the deploy dashboard. Deployments to production are frozen on Fridays after 2 PM to avoid weekend incidents.

# Code Review Policy

Every pull request must receive at least one approval from a senior engineer before it can be merged. Reviewers are expected to respond to review requests within one business day. Draft pull requests do not require review until marked ready.

# On-Call Rotation

The on-call rotation runs on a weekly basis, with each engineer taking a full week of primary on-call duty. The rotation schedule is published two weeks in advance. The primary on-call engineer is expected to acknowledge pages within 15 minutes during business hours.

# Incident Severity Levels

Incidents are classified as SEV1 (full outage), SEV2 (major degradation), or SEV3 (minor issue). SEV1 incidents require an immediate page to the entire engineering leadership team. A post-mortem is required for all SEV1 and SEV2 incidents within 48 hours of resolution.

# Branching Strategy

We use trunk-based development. Feature branches should be short-lived and merged into main within three days. Long-running feature branches are discouraged and require team lead approval.
