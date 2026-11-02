# 50 prompt smoke audit

Hand authored prompts and labels. Some identifier values may be synthetic, invalid, or out of policy; mismatches require manual review before becoming training data.

Models: dual-task checkpoint in `../best`; Laya intent checkpoint in `/Users/app/Downloads/Laya-Finetune/laya_finetuned`.
Dual threshold 0.5 for each binary intent; Laya threshold 0.7. Extraction uses all 45 trained labels and threshold 0.5. Laya has no extraction head.

Dual intent exact match against smoke expectations: 36/50. Laya: 35/50.
Dual extraction exact type and span: 8/40 positive prompts. This is a strict smoke check, not validated recall.
Five-label extraction chunks: 11/40 strict matches. Both schemas emitted no entities on the 10 negatives. Chunking changes the inference task and is not equivalent to published evaluation.
Models agree on intent flags: 45/50.

## By prompt category

| Category | Cases | Dual exact | Laya exact |
|---|---:|---:|---:|
| messy | 3 | 1 | 1 |
| near_miss | 2 | 2 | 2 |
| negative | 8 | 8 | 8 |
| positive | 37 | 25 | 24 |

## Observations

- Greeting and Sarah email classify correctly in both raw model APIs. No PCI false positives among the 10 negative/near-miss prompts in either model.
- Several synthetic PCI positives get no PCI flag from either model. Payment validity and dataset label policy must be checked before treating these as false negatives.
- Extraction is schema sensitive: all 45 labels produced no EMAIL for Sarah prompt, while EMAIL plus NAME labels produced EMAIL confidence 0.9996 in direct probe. Keep schema fixed when comparing with published evaluation.
- Previous UI PCI results came from entity-derived priority logic and a fabricated 0.99 confidence, not the trained classification output.
- Laya used a different intent training file (`laya_train_25.jsonl`) from the dual-task checkpoint (`v5_train/dual_train_100.jsonl`); they are related corpora but not identical training runs.
- Requested taxonomy includes aliases: TELEPHONE=PHONE, DRIVER_ID=DRIVING_LICENSE, IN_PERMANENT_ACCOUNT_NUMBER=PAN_NUMBER, IN_VOTER_NUMBER=VOTER_ID, CREDIT_DEBIT_NUMBER=CARD_NUMBER, CREDIT_DEBIT_CVV=CVV, CREDIT_DEBIT_EXPIRY=CARD_EXPIRY, BANK_ROUTING=US_ROUTING_NO. These are not separately trained types.

## Extraction cases needing review

| Type | Strict misses |
|---|---:|
| EMAIL | 2 |
| PHONE | 2 |
| CARD_NUMBER | 2 |
| NAME | 1 |
| ADDRESS | 1 |
| FAX | 1 |
| SSN | 1 |
| DRIVING_LICENSE | 1 |
| PASSPORT_NUMBER | 1 |
| VOTER_ID | 1 |
| PASSWORD | 1 |
| USERNAME | 1 |
| PAN_NUMBER | 1 |
| AADHAAR_NUMBER | 1 |
| CA_SOCIAL_INSURANCE_NUMBER | 1 |
| UK_NATIONAL_INSURANCE_NUMBER | 1 |
| US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER | 1 |
| IP_ADDRESS | 1 |
| MAC_ADDRESS | 1 |
| AWS_ACCESS_KEY | 1 |
| LICENSE_PLATE | 1 |
| VEHICLE_IDENTIFICATION_NUMBER | 1 |
| CERTIFICATE_LICENSE_NUMBER | 1 |
| URL | 1 |
| CVV | 1 |
| CARD_EXPIRY | 1 |
| ACCOUNT_NUMBER | 1 |
| US_ROUTING_NO | 1 |
| UTR_NUMBER | 1 |
