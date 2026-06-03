# Utah Next-10 Checklist Verification Smoke - 2026-06-03

## Setup
- Actor/planner: `gemma4:e4b` via Ollama `/api/chat` with `think:false`
- Fair condition: no web, no tools, no GPT teacher hint
- Method: Gemma4 first generates a text-only pathology visual checklist from question/options, then inspects the image and scores every option against that checklist

## Accuracy on Same 10 Hard Cases
- Original Gemma4+PRM baseline: 0/10 = 0.0%
- Generic multiview/crop: 1/10 = 10.0%
- Option-specific verification: 2/10 = 20.0%
- Model-generated checklist + verification: 1/10 = 10.0%
- GPT visual hint diagnostic: 4/10 = 40.0%
- GPT reference: 9/10 = 90.0%

## Sample-Level Comparison
| sample | gold | option_verify | checklist_verify | teacher_visual |
|---|---|---|---|---|
| utah_webpath_closed_1 | 0: Thrombocytopenia | 0: Thrombocytopenia OK | 2: Congestive heart failure NO | 0: Thrombocytopenia OK |
| utah_webpath_closed_10 | 2: Suicidal gunshot wound | 1: Accidental electrocution NO | 1: Accidental electrocution NO | 1: Accidental electrocution NO |
| utah_webpath_closed_16 | 0: Dysphagia | 4: Skin rash NO | 4: Skin rash NO | 4: Skin rash NO |
| utah_webpath_closed_41 | 5: Trisomy 21 | 4: Endocardial fibroelastosis NO | 4: Endocardial fibroelastosis NO | 5: Trisomy 21 OK |
| utah_webpath_closed_43 | 1: Intravenous drug use | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO | 3: Trousseau syndrome NO |
| utah_webpath_closed_45 | 3: Dense collagen bundles | 1: Granulomas with caseous necrosis NO | 1: Granulomas with caseous necrosis NO | 3: Dense collagen bundles OK |
| utah_webpath_closed_11 | 1: The mode (manner) of death is accident | 3: Breast carcinoma is a contributing cause o NO | 3: Breast carcinoma is a contributing cause o NO | 4: Pulmonary embolism is the underlying cause NO |
| utah_webpath_closed_32 | 4: A family history of an affected parent may not be present | 0: Expression of the defect is uniform among  NO | 0: Expression of the defect is uniform among  NO | 0: Expression of the defect is uniform among  NO |
| utah_webpath_closed_35 | 3: Respiratory distress | 3: Respiratory distress OK | 3: Respiratory distress OK | 3: Respiratory distress OK |
| utah_webpath_closed_0 | 3: Atherosclerosis | 0: Amyloidosis NO | 0: Amyloidosis NO | 4: Viral myocarditis NO |

## Takeaway
- Model-generated checklist verification scored 1/10, worse than plain option verification at 2/10.
- The checklist often contains useful expected findings, but Gemma4 still mis-scores visible evidence. Example: it generated petechiae/ecchymoses for thrombocytopenia, then assigned higher support to CHF/venous congestion on the same skin image.
- This suggests the next fair improvement should not be a generic pre-checklist from the same actor. Better candidates are: stricter scoring rules that penalize non-visual/systemic explanations, or a separate visual-descriptor subactor/model before option verification.
