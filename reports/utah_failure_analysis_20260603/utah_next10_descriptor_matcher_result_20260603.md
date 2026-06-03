# Utah Next-10 Descriptor + Matcher Smoke - 2026-06-03

## Setup
- Visual descriptor subactor: `gemma4:e4b`, sees image and clinical stem but not answer options
- Text matcher: `qwen2.5:7b-instruct`, sees descriptor, question, and options but not image
- Fair condition: no web, no tools, no GPT teacher hint

## Accuracy on Same 10 Hard Cases
- Original Gemma4+PRM baseline: 0/10 = 0.0%
- Option-specific verification: 2/10 = 20.0%
- Strict visual scoring: 2/10 = 20.0%
- Gemma4 descriptor + Qwen matcher: 1/10 = 10.0%
- GPT visual hint diagnostic: 4/10 = 40.0%
- GPT reference: 9/10 = 90.0%

## Sample-Level Comparison
| sample | gold | option_verify | strict_visual | descriptor_matcher | teacher_visual |
|---|---|---|---|---|---|
| utah_webpath_closed_1 | 0: Thrombocytopenia | 0: Thrombocytopenia OK | 0: Thrombocytopenia OK | 3: Metastatic breast carcinoma NO | 0: Thrombocytopenia OK |
| utah_webpath_closed_10 | 2: Suicidal gunshot wound | 1: Accidental electrocution NO | 1: Accidental electrocution NO | 3: Homicidal stab wounds NO | 1: Accidental electrocution NO |
| utah_webpath_closed_16 | 0: Dysphagia | 4: Skin rash NO | 4: Skin rash NO | 4: Skin rash NO | 4: Skin rash NO |
| utah_webpath_closed_41 | 5: Trisomy 21 | 4: Endocardial fibroelastosis NO | 4: Endocardial fibroelastosis NO | 4: Endocardial fibroelastosis NO | 5: Trisomy 21 OK |
| utah_webpath_closed_43 | 1: Intravenous drug use | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO | 0: Acute rheumatic fever NO | 3: Trousseau syndrome NO |
| utah_webpath_closed_45 | 3: Dense collagen bundles | 1: Granulomas with caseous necrosis NO | 4: Atypical squamous epithelium NO | 0: Necrotizing acute inflammation NO | 3: Dense collagen bundles OK |
| utah_webpath_closed_11 | 1: The mode (manner) of death is accident | 3: Breast carcinoma is a contributing cause o NO | 4: Pulmonary embolism is the underlying caus NO | 4: Pulmonary embolism is the underlying cause NO | 4: Pulmonary embolism is the underlying cause NO |
| utah_webpath_closed_32 | 4: A family history of an affected parent may not be present | 0: Expression of the defect is uniform among  NO | 0: Expression of the defect is uniform among  NO | 4: A family history of an affected parent may OK | 0: Expression of the defect is uniform among  NO |
| utah_webpath_closed_35 | 3: Respiratory distress | 3: Respiratory distress OK | 3: Respiratory distress OK | 2: Sepsis NO | 3: Respiratory distress OK |
| utah_webpath_closed_0 | 3: Atherosclerosis | 0: Amyloidosis NO | 0: Amyloidosis NO | 4: Viral myocarditis NO | 4: Viral myocarditis NO |

## Takeaway
- The separated descriptor/matcher pipeline scored 1/10, worse than direct option verification at 2/10.
- This means the current Gemma4 visual descriptor is not producing sufficiently discriminative pathology evidence when options are hidden.
- The positive teacher-hint result remains the strongest diagnostic signal: better visual evidence can help, but it likely needs a stronger/different visual descriptor model or training/distillation rather than only orchestration.
