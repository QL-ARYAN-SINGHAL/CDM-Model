# GLiNER2.5-Decide: intent + extract (fine-tuned)

## gold (300 rows)

| detector | P | R | F1 | best threshold (F1) |
|---|---|---|---|---|
| PII | 0.9487 | 0.9823 | 0.9652 | 0.05 |
| PHI | 1.0 | 1.0 | 1.0 | 0.05 |
| PCI | 0.973 | 0.9351 | 0.9536 | 0.75 |

exact-row accuracy 0.95; false-positive rate on None rows 0.0619

- extraction, all predicted types: P 0.9165 / R 0.893 / F1 0.9046; None-intent rows that still extract: 0.0973
- extraction, gated by own intent: P 0.931 / R 0.8791 / F1 0.9043; None-intent rows that still extract: 0.0619

| type | P | R | F1 | support |
|---|---|---|---|---|
| URL | 0.25 | 0.25 | 0.25 | 8 |
| USERNAME | 1.0 | 0.25 | 0.4 | 4 |
| CERTIFICATE_LICENSE_NUMBER | 1.0 | 0.4 | 0.5714 | 5 |
| PHONE | 1.0 | 0.4444 | 0.6154 | 9 |
| MAC_ADDRESS | 1.0 | 0.5 | 0.6667 | 4 |
| PAN_NUMBER | 0.5 | 1.0 | 0.6667 | 4 |
| SSN | 0.75 | 0.6 | 0.6667 | 5 |
| DATE | 1.0 | 0.5714 | 0.7273 | 7 |
| ACCOUNT_NUMBER | 0.8 | 0.7273 | 0.7619 | 11 |
| ADDRESS | 1.0 | 0.625 | 0.7692 | 8 |
| FAX | 0.7143 | 0.8333 | 0.7692 | 6 |
| UTR_NUMBER | 0.7143 | 0.8333 | 0.7692 | 6 |
| IP_ADDRESS | 1.0 | 0.6667 | 0.8 | 6 |
| AWS_ACCESS_KEY | 0.8889 | 0.8 | 0.8421 | 10 |
| DEVICE_IDENTIFIER | 1.0 | 0.7273 | 0.8421 | 11 |
| AGE | 1.0 | 0.75 | 0.8571 | 4 |
| DRIVING_LICENSE | 0.75 | 1.0 | 0.8571 | 3 |
| CARD_NUMBER | 0.8889 | 0.8889 | 0.8889 | 18 |
| IN_NREGA | 1.0 | 0.8 | 0.8889 | 5 |
| UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER | 1.0 | 0.8 | 0.8889 | 5 |
| AWS_SECRET_KEY | 1.0 | 0.8333 | 0.9091 | 12 |
| VOTER_ID | 0.8333 | 1.0 | 0.9091 | 5 |
| US_ROUTING_NO | 1.0 | 0.8571 | 0.9231 | 14 |
| PASSWORD | 1.0 | 0.875 | 0.9333 | 8 |
| MEDICAL_RECORD_NUMBER | 0.9444 | 0.9444 | 0.9444 | 18 |
| NAME | 0.9515 | 0.98 | 0.9655 | 100 |
| AADHAAR_NUMBER | 1.0 | 1.0 | 1.0 | 4 |
| CARD_EXPIRY | 1.0 | 1.0 | 1.0 | 15 |
| CA_HEALTH_NUMBER | 1.0 | 1.0 | 1.0 | 7 |
| CA_SOCIAL_INSURANCE_NUMBER | 1.0 | 1.0 | 1.0 | 6 |
| CVV | 1.0 | 1.0 | 1.0 | 9 |
| EMAIL | 1.0 | 1.0 | 1.0 | 6 |
| HEALTH_PLAN_BENEFICIARY_NUMBER | 1.0 | 1.0 | 1.0 | 11 |
| IFSC_CODE | 1.0 | 1.0 | 1.0 | 16 |
| INTERNATIONAL_BANK_ACCOUNT_NUMBER | 1.0 | 1.0 | 1.0 | 12 |
| LICENSE_PLATE | 1.0 | 1.0 | 1.0 | 2 |
| PASSPORT_NUMBER | 1.0 | 1.0 | 1.0 | 3 |
| PIN | 1.0 | 1.0 | 1.0 | 8 |
| SWIFT_CODE | 1.0 | 1.0 | 1.0 | 4 |
| UK_NATIONAL_HEALTH_SERVICE_NUMBER | 1.0 | 1.0 | 1.0 | 5 |
| UK_NATIONAL_INSURANCE_NUMBER | 1.0 | 1.0 | 1.0 | 4 |
| UPI_ID | 1.0 | 1.0 | 1.0 | 8 |
| US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER | 1.0 | 1.0 | 1.0 | 5 |
| VEHICLE_IDENTIFICATION_NUMBER | 1.0 | 1.0 | 1.0 | 3 |
| WALLET_ID | 1.0 | 1.0 | 1.0 | 6 |

## test (11406 rows)

| detector | P | R | F1 | best threshold (F1) |
|---|---|---|---|---|
| PII | 0.9876 | 0.9751 | 0.9813 | 0.1 |
| PHI | 0.994 | 0.9874 | 0.9907 | 0.05 |
| PCI | 0.9786 | 0.9092 | 0.9426 | 0.05 |

exact-row accuracy 0.9634; false-positive rate on None rows 0.0114

- extraction, all predicted types: P 0.9137 / R 0.8742 / F1 0.8935; None-intent rows that still extract: 0.0242
- extraction, gated by own intent: P 0.9301 / R 0.8511 / F1 0.8889; None-intent rows that still extract: 0.0096

| type | P | R | F1 | support |
|---|---|---|---|---|
| URL | 0.3296 | 0.3236 | 0.3266 | 275 |
| MAC_ADDRESS | 1.0 | 0.3836 | 0.5545 | 73 |
| ADDRESS | 0.9577 | 0.4689 | 0.6296 | 386 |
| SSN | 0.9493 | 0.5393 | 0.6878 | 382 |
| PASSWORD | 0.9333 | 0.56 | 0.7 | 25 |
| DRIVING_LICENSE | 0.9036 | 0.641 | 0.75 | 117 |
| LICENSE_PLATE | 0.8507 | 0.7037 | 0.7703 | 81 |
| CERTIFICATE_LICENSE_NUMBER | 0.853 | 0.7169 | 0.7791 | 332 |
| DEVICE_IDENTIFIER | 0.9016 | 0.7363 | 0.8106 | 311 |
| PHONE | 0.9135 | 0.7319 | 0.8127 | 332 |
| AADHAAR_NUMBER | 0.902 | 0.7419 | 0.8142 | 62 |
| FAX | 0.893 | 0.7614 | 0.822 | 285 |
| AGE | 0.9024 | 0.7708 | 0.8315 | 48 |
| IP_ADDRESS | 0.9449 | 0.7571 | 0.8406 | 317 |
| PAN_NUMBER | 0.875 | 0.8235 | 0.8485 | 85 |
| ACCOUNT_NUMBER | 0.9117 | 0.8126 | 0.8593 | 902 |
| AWS_SECRET_KEY | 1.0 | 0.7667 | 0.8679 | 30 |
| DATE | 0.9444 | 0.8213 | 0.8786 | 414 |
| MEDICAL_RECORD_NUMBER | 0.9039 | 0.8755 | 0.8895 | 763 |
| SWIFT_CODE | 0.9008 | 0.9083 | 0.9046 | 120 |
| US_ROUTING_NO | 0.9448 | 0.8725 | 0.9072 | 510 |
| EMAIL | 0.9493 | 0.87 | 0.9079 | 323 |
| PIN | 0.9091 | 0.9091 | 0.9091 | 33 |
| VEHICLE_IDENTIFICATION_NUMBER | 0.9268 | 0.8941 | 0.9102 | 85 |
| UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER | 0.9828 | 0.8507 | 0.912 | 67 |
| VOTER_ID | 0.9464 | 0.8833 | 0.9138 | 60 |
| CARD_NUMBER | 0.939 | 0.891 | 0.9143 | 587 |
| USERNAME | 1.0 | 0.8475 | 0.9174 | 59 |
| CARD_EXPIRY | 0.9106 | 0.9433 | 0.9266 | 529 |
| AWS_ACCESS_KEY | 0.9286 | 0.9286 | 0.9286 | 28 |
| UTR_NUMBER | 0.9427 | 0.9165 | 0.9294 | 395 |
| WALLET_ID | 0.9472 | 0.9288 | 0.9379 | 309 |
| CVV | 0.9396 | 0.9406 | 0.9401 | 909 |
| IN_NREGA | 1.0 | 0.8889 | 0.9412 | 36 |
| CA_SOCIAL_INSURANCE_NUMBER | 0.9861 | 0.9103 | 0.9467 | 78 |
| PASSPORT_NUMBER | 1.0 | 0.9 | 0.9474 | 50 |
| HEALTH_PLAN_BENEFICIARY_NUMBER | 0.9595 | 0.958 | 0.9588 | 619 |
| CA_HEALTH_NUMBER | 0.9744 | 0.9645 | 0.9694 | 197 |
| NAME | 0.9783 | 0.968 | 0.9731 | 1815 |
| IFSC_CODE | 0.9928 | 0.9674 | 0.9799 | 429 |
| UK_NATIONAL_HEALTH_SERVICE_NUMBER | 1.0 | 0.9697 | 0.9846 | 132 |
| INTERNATIONAL_BANK_ACCOUNT_NUMBER | 0.9935 | 0.9809 | 0.9872 | 157 |
| UPI_ID | 0.9953 | 0.9953 | 0.9953 | 430 |
| UK_NATIONAL_INSURANCE_NUMBER | 1.0 | 1.0 | 1.0 | 65 |
| US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER | 1.0 | 1.0 | 1.0 | 36 |

single-request latency: {'p50': 51.2, 'p95': 56.6, 'n': 100, 'device': 'cuda'}
