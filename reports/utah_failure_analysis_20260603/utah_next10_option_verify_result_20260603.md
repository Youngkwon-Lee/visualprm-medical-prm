# Utah Next-10 Option Verification Smoke - 2026-06-03

## Setup
- Actor: `gemma4:e4b` via Ollama `/api/chat` with `think:false`
- Dataset: the 10-sample debug set from Gemma4+PRM Utah failures
- Web/tool use: none
- `option_verify`: Gemma4 scores visible support/mismatch for every answer option
- `teacher_visual`: same as option_verify, but with GPT reference visual-observation sentences only; this is diagnostic, not a fair main benchmark

## Accuracy
- Original Gemma4+PRM baseline on these 10: 0/10 = 0.0%
- Generic multiview/crop controller: 1/10 = 10.0%
- Option-specific visual verification: 2/10 = 20.0%
- Teacher visual hint + option verification: 4/10 = 40.0%
- GPT reference: 9/10 = 90.0%

## Sample-Level Comparison
| sample | gold | baseline | option_verify | teacher_visual | GPT ref |
|---|---|---|---|---|---|
| utah_webpath_closed_1 | 0: Thrombocytopenia | 4: Pellagra NO | 0: Thrombocytopenia OK | 0: Thrombocytopenia OK | 0: Thrombocytopenia OK |
| utah_webpath_closed_10 | 2: Suicidal gunshot wound | 5: Fall from a height NO | 1: Accidental electrocution NO | 1: Accidental electrocution NO | 2: Suicidal gunshot wound OK |
| utah_webpath_closed_16 | 0: Dysphagia | 4: Skin rash NO | 4: Skin rash NO | 4: Skin rash NO | 0: Dysphagia OK |
| utah_webpath_closed_41 | 5: Trisomy 21 | 4: Endocardial fibroelastosis NO | 4: Endocardial fibroelastosis NO | 5: Trisomy 21 OK | 5: Trisomy 21 OK |
| utah_webpath_closed_43 | 1: Intravenous drug use | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO | 1: Intravenous drug use OK |
| utah_webpath_closed_45 | 3: Dense collagen bundles | 1: Granulomas with caseous necrosis NO | 1: Granulomas with caseous necrosis NO | 3: Dense collagen bundles OK | 3: Dense collagen bundles OK |
| utah_webpath_closed_11 | 1: The mode (manner) of death is accident | 4: Pulmonary embolism is the underlying cause NO | 3: Breast carcinoma is a contributing cause o NO | 4: Pulmonary embolism is the underlying cause NO | 1: The mode (manner) of death is accident OK |
| utah_webpath_closed_32 | 4: A family history of an affected parent may not be present | 1: {"sample_id": "utah_webpath_closed_32_pp00 NO | 0: Expression of the defect is uniform among  NO | 0: Expression of the defect is uniform among  NO | 4: A family history of an affected parent may OK |
| utah_webpath_closed_35 | 3: Respiratory distress | 0: Intraventricular hemorrhage NO | 3: Respiratory distress OK | 3: Respiratory distress OK | 3: Respiratory distress OK |
| utah_webpath_closed_0 | 3: Atherosclerosis | 0: Amyloidosis NO | 0: Amyloidosis NO | 4: Viral myocarditis NO | 4: Viral myocarditis NO |

## Takeaway
- Option-specific verification is more stable than the generic multiview controller: valid 10/10 and 2/10 correct on the hard debug set.
- Teacher visual hints improve this to 4/10, which means the actor can sometimes choose correctly once the right visible evidence is surfaced.
- The bottleneck is still visual evidence extraction, not just final answer selection. The next fair improvement should generate better visual inventories without using GPT teacher hints, e.g. pathology-specific visual checklist prompts or a stronger small VLM as a visual-descriptor subactor.
