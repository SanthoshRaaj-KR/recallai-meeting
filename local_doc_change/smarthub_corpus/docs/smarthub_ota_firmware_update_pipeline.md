# OTA Firmware Update Pipeline — Technical Spec

INFER Manage supports secure over-the-air firmware updates for 2,000+ device types. This page covers the full pipeline from campaign creation through delivery, verification, and rollback.

## Update Campaign State Machine


    |  State | Description | Transitions |

    |  DRAFT | Campaign created, not yet started | → STAGED (on start) | → CANCELLED |

    |  STAGED | Firmware package uploaded and validated; ring 0 (5%) selected | → RING_0_ACTIVE |

    |  RING_0_ACTIVE | Deploying to 5% of target devices | → RING_0_BAKING | → ROLLING_BACK (failure > 5%) |

    |  RING_0_BAKING | Observing ring 0 for 2 hours post-update | → RING_1_ACTIVE | → ROLLING_BACK |

    |  RING_1_ACTIVE | Deploying to 25% of remaining devices | → RING_1_BAKING | → ROLLING_BACK |

    |  RING_1_BAKING | Bake period: 4 hours | → FULL_ROLLOUT | → ROLLING_BACK |

    |  FULL_ROLLOUT | Deploying to 100% of remaining devices | → COMPLETED | → PARTIAL_FAILURE |

    |  COMPLETED | All devices updated successfully | Terminal |

    |  ROLLING_BACK | Reverting ring 0/1 devices to previous firmware | → ROLLED_BACK | → ROLLBACK_FAILED |

    |  PARTIAL_FAILURE | Full rollout complete but >2% devices failed | → COMPLETED (after manual override) |



## Firmware Package Format

bash

## manifest.json Schema

json

## Delivery Protocol

  - Campaign scheduler (Go service) selects target devices for current ring using consistent hashing on `device_id`
  - SD-EDGE Gateway receives update job via MQTT topic `infer/gw/{gateway_id}/ota/command`
  - Gateway downloads firmware package from pre-signed S3 URL (1-hour TTL) over HTTPS
  - Gateway verifies: SHA-256 hash of downloaded file, then ECDSA signature against embedded public key
  - Gateway pushes firmware to device using device-class-specific protocol (ONVIF firmware upgrade API, SSH SCP, TFTP, HTTP PUT)
  - Device reboots; gateway polls for reconnection (timeout: 10 min)
  - Gateway reports result to INFER cloud via MQTT `infer/gw/{gateway_id}/ota/result`

## Rollback Logic

python RingDecision:
    stats = get_ring_stats(campaign_id, ring)
    failure_rate = stats.failed / stats.attempted

    if failure_rate > ROLLBACK_THRESHOLD:
        logger.warning(
            "Ring %d failure rate %.1f%% exceeds threshold — triggering rollback",
            ring, failure_rate * 100,
        )
        return RingDecision.ROLLBACK

    # Also check health metrics post-update
    health = get_post_update_health(campaign_id, ring)
    if health.avg_sps_delta 15 points
        return RingDecision.ROLLBACK

    return RingDecision.PROCEED]]>

## Bandwidth Optimization


    |  Technique | Saving | Applicability |

    |  Binary delta patches (bsdiff) | 60–85% reduction | When previous version is known and delta exists |

    |  SD-EDGE Gateway P2P sharing | Eliminates redundant S3 downloads within a site | When multiple devices of same model at same site |

    |  Scheduled maintenance windows | Avoids business-hours bandwidth impact | All campaigns |

    |  Parallel update throttle | Max 10 concurrent updates per gateway | Prevents gateway CPU saturation |
