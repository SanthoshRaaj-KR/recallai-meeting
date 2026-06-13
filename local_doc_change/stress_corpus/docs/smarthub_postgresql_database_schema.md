# PostgreSQL Database Schema — Core Tables

INFER uses PostgreSQL 15 (AWS RDS Multi-AZ) as the primary relational store. Tenant isolation uses a schema-per-tenant model. This page documents core tables in the `infer_shared` schema (cross-tenant) and the per-tenant schema.

## infer_shared — Cross-Tenant Tables

sql= 7.0;]]>

## Per-Tenant Schema — Core Tables

sql

## Partitioning Strategy


    |  Table | Partition By | Retention | Archive |

    |  device_telemetry_hourly | RANGE on hour (monthly partitions) | 13 months hot | S3 Parquet via pg_partman + custom archiver |

    |  alerts | RANGE on created_at (quarterly) | 3 years hot | S3 after 3 years |

    |  compliance_checks | RANGE on checked_at (monthly) | 1 year hot | S3 after 1 year |

    |  audit_log | RANGE on ts (monthly) | 7 years (immutable) | Glacier Deep Archive |



## Connection Pooling

PgBouncer runs as a sidecar in each API service pod, configured in transaction-mode pooling:


    |  Parameter | Value |

    |  pool_mode | transaction |

    |  max_client_conn | 1000 per API replica |

    |  default_pool_size | 25 |

    |  reserve_pool_size | 5 |

    |  server_idle_timeout | 600 s |

    |  RDS max_connections | 5000 (db.r6g.4xlarge) |
