# Authentication & Authorization — Technical Deep Dive

INFER uses a layered auth model: Auth0 for identity brokering, short-lived JWTs for API access, Open Policy Agent (OPA) for fine-grained authorization, and per-device mTLS for gateway-to-cloud channels.

## JWT Token Anatomy

json

## RBAC Role Definitions


    |  Role | Permissions | Typical Assignee |

    |  org_admin | All permissions + user management + billing | IT Director / CISO |

    |  security_analyst | Read all; acknowledge/resolve alerts; quarantine devices; generate reports | SOC Analyst |

    |  device_manager | Read all; onboard/decommission devices; trigger OTA; edit device metadata | IT Admin / NOC Engineer |

    |  compliance_auditor | Read-only: devices, compliance, reports; export PDF | Internal Auditor / GRC |

    |  alert_viewer | Read-only: alerts, device health, dashboards | Help Desk / L1 Support |

    |  api_integration | Scoped API access per integration; no UI login | SIEM / SOAR service account |



## OPA Policy Example — Device Quarantine

rego

## mTLS: Gateway-to-Cloud Channel

Every SD-EDGE Gateway authenticates to the INFER cloud using mutual TLS with a per-gateway X.509 certificate issued by SmartHub's private CA (Mocana CMS-backed).


    |  Parameter | Value |

    |  CA hierarchy | Root CA (offline HSM) → Intermediate CA → Gateway leaf cert |

    |  Leaf cert lifetime | 365 days; auto-rotated 30 days before expiry via ACME-like renewal |

    |  Key algorithm | ECDSA P-256 |

    |  TLS version | TLS 1.3 only; TLS 1.2 disabled |

    |  Cipher suites | TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256 |

    |  Certificate pinning | Gateway pins intermediate CA SPKI hash; rejects if mismatch |

    |  Revocation | OCSP Stapling; gateway checks every 4 hours |



## SAML 2.0 SSO Flow

  - User navigates to `https://app.smarthub.ai`
  - INFER redirects to Auth0 with `connection=saml-<tenant_slug>`
  - Auth0 sends SAML AuthnRequest to customer IdP (Okta / Azure AD / Ping)
  - IdP authenticates user; returns signed SAML assertion
  - Auth0 validates assertion; maps IdP group claims to INFER roles via attribute mapping config
  - Auth0 issues INFER JWT (RS256, 1-hour TTL) + refresh token (rotating, 7-day TTL)
  - Frontend stores JWT in memory only (no localStorage); refresh token in httpOnly cookie

## Token Revocation

Token revocation is maintained in Redis (cluster mode, 3 shards). On logout or account suspension:

python
