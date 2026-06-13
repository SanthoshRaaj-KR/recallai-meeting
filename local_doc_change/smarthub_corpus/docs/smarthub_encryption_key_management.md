# Encryption & Key Management

SmartHub.ai follows a defense-in-depth encryption strategy covering data at rest, data in transit, application-layer secrets, and device credentials. This page is the single authoritative reference for all encryption choices.

## Data at Rest


    |  Data Store | Encryption | Key Management |

    |  PostgreSQL (RDS) | AES-256 (AWS storage encryption) | AWS KMS CMK, per-tenant key |

    |  S3 (telemetry archive) | SSE-KMS (AES-256) | AWS KMS CMK, per-tenant key |

    |  Redis (ElastiCache) | AES-256 (encryption at rest enabled) | AWS KMS managed key |

    |  InfluxDB Cloud | AES-256 (InfluxData-managed) | InfluxData KMS |

    |  MSK (Kafka) | AES-256 (broker-level) | AWS KMS managed key |

    |  Neo4j AuraDB | AES-256 (Neo4j-managed) | Neo4j internal KMS |

    |  Secrets (API keys, tokens) | AES-256-GCM (application-layer) | AWS Secrets Manager with automatic 90-day rotation |



## Data in Transit


    |  Channel | Protocol | Minimum TLS | Certificate |

    |  Client → API Gateway | HTTPS | TLS 1.2 (1.3 preferred) | Let's Encrypt via ACM |

    |  API Gateway → ECS | HTTPS | TLS 1.2 | ACM private CA |

    |  ECS → PostgreSQL | PostgreSQL TLS | TLS 1.2 | RDS-managed cert |

    |  ECS → MSK | Kafka TLS | TLS 1.2 | ACM private CA |

    |  SD-EDGE Gateway → INFER cloud | mTLS over HTTPS/MQTT | TLS 1.3 only | Per-gateway cert (Mocana CA) |

    |  Gateway → Device (SNMP v3) | SNMP v3 authPriv | AES-128 (privacy) | Pre-shared key |

    |  Gateway → Device (OTA/SSH) | SSH | ED25519 host key | Per-device key stored in Secrets Manager |



## Application-Layer Sensitive Field Encryption

Selected columns in PostgreSQL are encrypted at the application layer before write (envelope encryption), in addition to RDS storage encryption:

python str:
    dek = get_tenant_dek(tenant_id)         # 256-bit data encryption key
    iv = os.urandom(12)                     # 96-bit IV for GCM
    cipher = AES.new(dek, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    # Store as base64(iv || tag || ciphertext)
    return b64encode(iv + tag + ciphertext).decode()]]>

## Key Hierarchy


    |  Level | Key Type | Lifetime | Storage |

    |  Master Key (KMK) | AWS KMS CMK (RSA-4096) | Permanent (auto-rotate annually) | AWS KMS HSM |

    |  Data Encryption Key (DEK) | AES-256-GCM, per-tenant | 90 days (auto-rotate) | AWS Secrets Manager (encrypted by KMK) |

    |  Session Keys (TLS) | ECDHE P-256 (ephemeral) | Per connection (forward secrecy) | Memory only |

    |  Gateway Identity Cert | ECDSA P-256 | 365 days (auto-renew at 30d) | Gateway TPM / Secrets Manager |

    |  JWT Signing Key | RSA-2048 (Auth0-managed) | 30 days (automatic rotation) | Auth0 JWKS endpoint |



## Crypto Deprecation Schedule


    |  Algorithm | Status | Deprecation Date | Migration Path |

    |  TLS 1.0 / 1.1 | Already disabled | 2022-01-01 | TLS 1.2+ |

    |  RSA-1024 device certs | Already disabled | 2023-06-01 | ECDSA P-256 |

    |  SHA-1 signatures | Already disabled | 2023-01-01 | SHA-256+ |

    |  AES-128-CBC (SNMP privacy) | Deprecating | 2025-01-01 | AES-256-CFB (SNMP v3) |

    |  RSA-2048 JWT signing | Reviewing | TBD (post-NIST PQC standards) | ML-KEM / ML-DSA (post-quantum) |
