# Utah Next-10 Strict Visual Scoring Smoke - 2026-06-03

## Setup
- Actor: `gemma4:e4b` via Ollama `/api/chat` with `think:false`
- Fair condition: no web, no tools, no GPT teacher hint
- Method: option-specific verification with strict score caps: direct visual support is required for score 4-5; stem-only support is capped at 2; general plausibility is capped at 1

## Accuracy on Same 10 Hard Cases
- Original Gemma4+PRM baseline: 0/10 = 0.0%
- Option-specific verification: 2/10 = 20.0%
- Strict visual scoring: 2/10 = 20.0%
- Model-generated checklist + verification: 1/10 = 10.0%
- GPT visual hint diagnostic: 4/10 = 40.0%
- GPT reference: 9/10 = 90.0%

## Sample-Level Comparison
| sample | gold | option_verify | strict_visual | teacher_visual |
|---|---|---|---|---|
| utah_webpath_closed_1 | 0: Thrombocytopenia | 0: Thrombocytopenia OK | 0: Thrombocytopenia OK | 0: Thrombocytopenia OK |
| utah_webpath_closed_10 | 2: Suicidal gunshot wound | 1: Accidental electrocution NO | 1: Accidental electrocution NO | 1: Accidental electrocution NO |
| utah_webpath_closed_16 | 0: Dysphagia | 4: Skin rash NO | 4: Skin rash NO | 4: Skin rash NO |
| utah_webpath_closed_41 | 5: Trisomy 21 | 4: Endocardial fibroelastosis NO | 4: Endocardial fibroelastosis NO | 5: Trisomy 21 OK |
| utah_webpath_closed_43 | 1: Intravenous drug use | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO |
| utah_webpath_closed_45 | 3: Dense collagen bundles | 1: Granulomas with caseous necrosis NO | 4: Atypical squamous epithelium NO | 3: Dense collagen bundles OK |
| utah_webpath_closed_11 | 1: The mode (manner) of death is accident | 3: Breast carcinoma is a contributing cause o NO | 4: Pulmonary embolism is the underlying caus NO | 4: Pulmonary embolism is the underlying cause NO |
| utah_webpath_closed_32 | 4: A family history of an affected parent may not be present | 0: Expression of the defect is uniform among  NO | 0: Expression of the defect is uniform among  NO | 0: Expression of the defect is uniform among  NO |
| utah_webpath_closed_35 | 3: Respiratory distress | 3: Respiratory distress OK | 3: Respiratory distress OK | 3: Respiratory distress OK |
| utah_webpath_closed_0 | 3: Atherosclerosis | 0: Amyloidosis NO | 0: Amyloidosis NO | 4: Viral myocarditis NO |

## Takeaway
- Strict visual scoring matched plain option verification at 2/10. It did not close the gap to teacher visual hints at 4/10.
- The rule helps keep output valid and discourages some non-visual explanations, but Gemma4 still often extracts the wrong visual inventory or maps findings to the wrong pathology.
- This supports the current hypothesis: the next real bottleneck is a stronger or separate visual-descriptor subactor, not more controller-side scoring prompts from the same Gemma4 actor.
