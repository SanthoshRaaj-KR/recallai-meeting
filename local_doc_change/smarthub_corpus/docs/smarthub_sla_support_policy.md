# SLA & Support Policy

This document defines SmartHub.ai's service level agreements (SLAs) and support procedures for INFER™ SaaS customers. Effective: January 1, 2024.

## Service Availability SLAs


    |  Component | Standard | Professional | Enterprise |

    |  INFER Core API | 99.5% | 99.9% | 99.95% |

    |  INFER Dashboard (Web UI) | 99.5% | 99.9% | 99.95% |

    |  Telemetry Ingestion Pipeline | 99.0% | 99.5% | 99.9% |

    |  Alert Notification Delivery | Best effort | 99.0% | 99.5% |



*Availability calculated monthly, excluding scheduled maintenance windows (communicated 72h in advance via status.smarthub.ai).*

## Support Tiers


    |  Severity | Definition | Standard Response | Professional Response | Enterprise Response |

    |  P1 — Critical | Platform completely unavailable; active security breach in progress | 4 hours | 1 hour | 15 minutes (24/7 pager) |

    |  P2 — High | Major feature impaired; significant impact on security operations | 8 business hours | 4 business hours | 1 business hour |

    |  P3 — Medium | Minor feature impaired; workaround available | 2 business days | 1 business day | 4 business hours |

    |  P4 — Low | Feature request, general question, cosmetic issue | 5 business days | 3 business days | 2 business days |



## Support Channels


    |  Channel | Standard | Professional | Enterprise |

    |  In-app Help & Diagnostics | Yes | Yes | Yes |

    |  Knowledge Base | Yes | Yes | Yes |

    |  Email ([support@smarthub.ai](mailto:support@smarthub.ai)) | Yes (business hours) | Yes (business hours) | Yes (24/7) |

    |  Slack Connect | No | No | Yes (dedicated channel) |

    |  Phone | No | No | Yes (P1/P2 only) |

    |  Technical Account Manager (TAM) | No | No | Yes (named TAM) |



## Scheduled Maintenance

  - Maintenance windows: Sundays 02:00–06:00 UTC
  - Advanced notice: at least 72 hours for planned maintenance; 24 hours for urgent security patches
  - Notification channels: [status.smarthub.ai](https://status.smarthub.ai), email to account admin, in-app banner
  - Emergency security patches: may be deployed with 2-hour notice; downtime capped at 30 minutes

## Incident Communication Process

  - Incident detected (automated monitoring or customer report)
  - Status page updated within 15 minutes of confirmed incident
  - Email notification to all affected tenant admins
  - Updates posted every 30 minutes during active P1 incidents
  - Post-incident report (RCA) published within 5 business days of resolution

## SLA Credit Policy

If monthly availability falls below the contracted SLA, customers are eligible for service credits:


    |  Availability Achieved | Credit (% of monthly fee) |

    |  Below SLA but >99.0% | 10% |

    |  98.0% – 99.0% | 25% |

    |  95.0% – 97.9% | 50% |

    |  Below 95.0% | 100% |
