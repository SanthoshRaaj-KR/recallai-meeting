# API Reference Guide — INFER REST API v2

Base URL: `https://api.smarthub.ai/v2`

Authentication: Bearer token (JWT) obtained via `POST /auth/token`. All requests must include `Authorization: Bearer <token>`.

## Authentication


    |  Method | Endpoint | Description |

    |  POST | /auth/token | Exchange API key for JWT access token (expires 1h) |

    |  POST | /auth/refresh | Refresh access token using refresh token |

    |  DELETE | /auth/token | Revoke current token |



## Devices


    |  Method | Endpoint | Description |

    |  GET | /devices | List all devices (paginated). Query params: `page`, `limit`, `site_id`, `status`, `device_class` |

    |  GET | /devices/{device_id} | Get full device record including SPS, last seen, firmware version |

    |  POST | /devices | Manually register a device (for devices not auto-discovered) |

    |  PATCH | /devices/{device_id} | Update device metadata (label, site assignment, group tags) |

    |  DELETE | /devices/{device_id} | Decommission a device (removes from active inventory) |

    |  POST | /devices/bulk-import | CSV bulk import; returns job_id for async status polling |

    |  GET | /devices/{device_id}/telemetry | Historical telemetry. Query params: `start`, `end`, `metric` |

    |  POST | /devices/{device_id}/quarantine | Isolate device on the network immediately |

    |  POST | /devices/{device_id}/ota-update | Trigger OTA firmware update; returns job_id |



## Alerts


    |  Method | Endpoint | Description |

    |  GET | /alerts | List alerts. Filter by `severity`, `status`, `device_id`, `site_id` |

    |  GET | /alerts/{alert_id} | Full alert detail including evidence and recommended actions |

    |  PATCH | /alerts/{alert_id} | Update alert status (`acknowledged`, `resolved`, `false_positive`) |

    |  GET | /alerts/summary | Aggregated alert counts by severity / site / device class |



## Compliance


    |  Method | Endpoint | Description |

    |  GET | /compliance/posture | Tenant-wide compliance posture summary per framework |

    |  GET | /compliance/posture/{device_id} | Per-device compliance detail (NIST CSF, IEC 62443, etc.) |

    |  POST | /compliance/report | Generate downloadable PDF compliance report |

    |  GET | /compliance/policies | List active compliance policies for tenant |

    |  POST | /compliance/policies | Create custom compliance policy rule |



## Example: List Critical Alerts

bash

## Rate Limits


    |  Tier | Requests / Minute | Burst |

    |  Free / Trial | 60 | 10 |

    |  Standard | 600 | 100 |

    |  Enterprise | 6,000 | 1,000 |
