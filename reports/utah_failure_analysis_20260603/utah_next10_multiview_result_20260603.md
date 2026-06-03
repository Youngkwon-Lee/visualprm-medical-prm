# Utah Next-10 Multiview Smoke - 2026-06-03

Input: `analysis/utah_next10_failure_debug_for_openclaw.json`
Output: `results_native_openclaw/utah_next10_failure_gemma4_multiview_chat_thinkfalse_20260603.jsonl`

## Result
- Final valid: 9/10
- Final correct: 1/9 valid = 11.11%
- Earlier `/api/generate` run produced mostly empty responses because `gemma4:e4b` is a thinking-capable Ollama model. Switching to `/api/chat` with `think:false` fixed output stability.

## Branch Summary
| branch | valid | correct |
|---|---:|---:|
| multiview:original | 6/10 | 0/10 |
| multiview:contrast_sharpen | 4/10 | 0/10 |
| multiview:center_zoom | 7/10 | 1/10 |
| option_check:original | 5/10 | 0/10 |
| controlled_knowledge:original | 3/10 | 0/10 |

## Final Predictions
| sample | gold | final | valid | correct | votes |
|---|---|---|---:|---:|---|
| utah_webpath_closed_1 | 0: Thrombocytopenia | None: None | False | False | {} |
| utah_webpath_closed_10 | 2: Suicidal gunshot wound | 1: Accidental electrocution | True | False | {'1': 5} |
| utah_webpath_closed_16 | 0: Dysphagia | 4: Skin rash | True | False | {'4': 3} |
| utah_webpath_closed_41 | 5: Trisomy 21 | 4: Endocardial fibroelastosis | True | False | {'4': 3} |
| utah_webpath_closed_43 | 1: Intravenous drug use | 3: Trousseau syndrome | True | False | {'3': 3} |
| utah_webpath_closed_45 | 3: Dense collagen bundles | 1: Granulomas with caseous necrosis | True | False | {'1': 2} |
| utah_webpath_closed_11 | 1: The mode (manner) of death is accident | 4: Pulmonary embolism is the underlying cause of death | True | False | {'4': 2} |
| utah_webpath_closed_32 | 4: A family history of an affected parent may not be present | 1: Heterozygous females rarely express the full phenotypic chan | True | False | {'1': 1} |
| utah_webpath_closed_35 | 3: Respiratory distress | 3: Respiratory distress | True | True | {'3': 1} |
| utah_webpath_closed_0 | 3: Atherosclerosis | 0: Amyloidosis | True | False | {'0': 5} |

## Correct Individual Attempts
| sample | branch | answer |
|---|---|---|
| utah_webpath_closed_35 | multiview:center_zoom | 3: Respiratory distress |

## Takeaway
This simple multiview/contrast/center-crop majority controller is not enough. It improves output stability after `think:false`, but it does not fix Gemma4/e4b visual grounding on the hard Utah cases. The next useful direction is not more generic crops; it is option-specific visual verification or teacher-guided visual evidence distillation.
