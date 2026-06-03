# Utah WebPath Closed Failure Analysis - 2026-06-03

Baseline analyzed: Gemma4/e4b vision-direct, 8 candidates, MedVisualPRM Branch B rerank, no image tool, no web tool.

## Files
- PRM result: `/Users/youngkwon/projects/visualprm_openclaw_harness/results_native_openclaw/utah_webpath_closed_gemma4_vision_direct_prm8_0_97_prm.jsonl`
- Candidate result: `/Users/youngkwon/projects/visualprm_openclaw_harness/results_native_openclaw/utah_webpath_closed_gemma4_vision_direct_prm8_0_97_candidates.jsonl`
- Strong VLM reference: `/Users/youngkwon/projects/visualprm_openclaw_harness/results_native_openclaw/utah_webpath_closed_openai_gpt54_direct_0_97_newkey.jsonl`
- CSV output: `/Users/youngkwon/projects/visualprm_openclaw_harness/analysis/utah_gemma4_prm8_failure_analysis_20260603.csv`

## Main Counts
- Final PRM accuracy: 59/97 = 60.82%
- Candidate-stage final accuracy: 55/97 = 56.70%
- GPT reference accuracy: 86/97 = 88.66%
- Gemma+PRM wrong cases: 38
- Wrong cases where GPT reference is correct: 30/38
- Wrong cases where one of 8 Gemma candidates contained the gold answer: 9/38

## Failure Categories
- `actor_visual_grounding_or_lowB_miss`: 21
- `prm_rerank_missed_gold_candidate`: 9
- `hard_or_ambiguous_even_strong_vlm`: 7
- `needs_gpt_reference_rerun`: 1

Interpretation: most failures are not because the dataset is impossible. In 30 of 38 wrong cases, the strong VLM reference got the item right, so the main bottleneck is Gemma4/e4b visual grounding or low-B reasoning. PRM also missed 9 cases where a correct candidate existed among the 8 generated answers.

## PRM Calibration Check
- Correct final PRM scores: n=59, mean=0.762, median=0.788, min=0.241, max=0.996
- Wrong final PRM scores: n=38, mean=0.753, median=0.778, min=0.417, max=0.990

Interpretation: PRM score does not separate correct and incorrect answers well in this run. The wrong-answer mean and median are close to the correct-answer mean and median, so simply increasing the PRM threshold is unlikely to solve the issue.

## PRM Rerank Misses
| sample | gold | PRM chose | gold votes in 8 | candidate votes |
|---|---|---|---:|---|
| utah_webpath_closed_11 | 1: The mode (manner) of death is accident | 4: Pulmonary embolism is the underlying cause of death. | 1 | 0:4; 4:3; 1:1 |
| utah_webpath_closed_32 | 4: A family history of an affected parent may not be present | 1: {"sample_id": "utah_webpath_closed_32_pp003.jpg", "steps": ["The image shows a b | 1 | 1:6; 2:1; 4:1 |
| utah_webpath_closed_35 | 3: Respiratory distress | 0: Intraventricular hemorrhage | 5 | 3:5; 0:3 |
| utah_webpath_closed_37 | 0: Hemorrhage | 2: Stillbirth | 6 | 0:6; 2:2 |
| utah_webpath_closed_39 | 4: Incidental finding | 3: Greater hemorrhage at delivery | 2 | 0:3; 3:2; 4:2; 2:1 |
| utah_webpath_closed_54 | 1: She will probably survive for at least 10 years | 2: Anti-microsomal and anti-thyroglobulin antibodies are detectable | 1 | 2:5; 4:2; 1:1 |
| utah_webpath_closed_55 | 2: Chronic alcohol use | 5: Alpha-1-antitrypsin deficiency | 7 | 2:7; 5:1 |
| utah_webpath_closed_70 | 3: Osteoporosis | 4: Multiple myeloma | 5 | 3:5; 4:2; 5:1 |
| utah_webpath_closed_89 | 4: Focal segmental glomerulosclerosis | 1: Nodular glomerulosclerosis | 7 | 4:7; 1:1 |

## Actor/Visual Grounding Misses
| sample | gold | Gemma+PRM chose | GPT chose | candidate votes |
|---|---|---|---|---|
| utah_webpath_closed_1 | 0: Thrombocytopenia | 4: Pellagra | 0: Thrombocytopenia | 4:8 |
| utah_webpath_closed_10 | 2: Suicidal gunshot wound | 5: Fall from a height | 2: Suicidal gunshot wound | 3:4; 5:3; 1:1 |
| utah_webpath_closed_16 | 0: Dysphagia | 4: Skin rash | 0: Dysphagia | 4:8 |
| utah_webpath_closed_41 | 5: Trisomy 21 | 4: Endocardial fibroelastosis | 5: Trisomy 21 | 4:8 |
| utah_webpath_closed_43 | 1: Intravenous drug use | 3: Trousseau syndrome | 1: Intravenous drug use | 3:4; 2:3 |
| utah_webpath_closed_45 | 3: Dense collagen bundles | 1: Granulomas with caseous necrosis | 3: Dense collagen bundles | 1:8 |
| utah_webpath_closed_47 | 1: Thickened squamous epithelium with keratin-filled cysts | 0: Atypical cells containing melanin pigment | 1: Thickened squamous epithelium with keratin-filled cysts | 0:8 |
| utah_webpath_closed_56 | 3: Celiac disease | 2: Celiac disease | 3: Celiac disease | 2:8 |
| utah_webpath_closed_61 | 0: Ketone bodies in urine | 1: Culture with Staphylococcus aureus | 0: Ketone bodies in urine | 1:8 |
| utah_webpath_closed_65 | 4: Hypergammaglobulinemia | 2: Decreased leukocyte alkaline phosphatase | 4: Hypergammaglobulinemia | 2:8 |
| utah_webpath_closed_67 | 6: Chondrosarcoma | 1: Osteosarcoma | 6: Chondrosarcoma | 1:6; 2:1; 3:1 |
| utah_webpath_closed_68 | 4: Metastatic carcinoma | 0: Paget disease of bone | 4: Metastatic carcinoma | 0:8 |

## Hard / Ambiguous or Reference-Disagrees
| sample | gold | Gemma+PRM chose | GPT chose |
|---|---|---|---|
| utah_webpath_closed_0 | 3: Atherosclerosis | 0: Amyloidosis | 4: Viral myocarditis |
| utah_webpath_closed_3 | 1: Diabetes mellitus, type I | 0: Marfan syndrome | 4: Aging |
| utah_webpath_closed_7 | 1: Acute meningitis | 2: Liquefactive necrosis | 2: Liquefactive necrosis |
| utah_webpath_closed_23 | 2: Thromboxane | 1: Leukotriene B4 | 3: TGF-beta |
| utah_webpath_closed_24 | 3: Transudation | 1: Erythema | 0: Exudation |
| utah_webpath_closed_33 | 0: Maternal serum alpha-fetoprotein 2 MoM | 2: Elevated serum hemoglobin A1C | 2: Elevated serum hemoglobin A1C |
| utah_webpath_closed_66 | 5: Metastatic carcinoma | 3: Myelofibrosis | 3: Myelofibrosis |
| utah_webpath_closed_8 | 0: Apoptosis | 2: Coagulative necrosis | GPT invalid: rerun needed |

## Recommended Next 10-Sample Debug Set
Use these before another full 97 run. They cover actor visual misses, PRM rerank misses, and one hard/reference-disagree case.

| sample | category | next action |
|---|---|---|
| utah_webpath_closed_1 | actor_visual_grounding_or_lowB_miss | try_roi_multiview_or_stronger_visual_actor |
| utah_webpath_closed_10 | actor_visual_grounding_or_lowB_miss | try_roi_multiview_or_stronger_visual_actor |
| utah_webpath_closed_16 | actor_visual_grounding_or_lowB_miss | try_roi_multiview_or_stronger_visual_actor |
| utah_webpath_closed_41 | actor_visual_grounding_or_lowB_miss | try_roi_multiview_or_stronger_visual_actor |
| utah_webpath_closed_43 | actor_visual_grounding_or_lowB_miss | try_roi_multiview_or_stronger_visual_actor |
| utah_webpath_closed_45 | actor_visual_grounding_or_lowB_miss | try_roi_multiview_or_stronger_visual_actor |
| utah_webpath_closed_11 | prm_rerank_missed_gold_candidate | improve_prm_calibration_or_answer_aggregation |
| utah_webpath_closed_32 | prm_rerank_missed_gold_candidate | improve_prm_calibration_or_answer_aggregation |
| utah_webpath_closed_35 | prm_rerank_missed_gold_candidate | improve_prm_calibration_or_answer_aggregation |
| utah_webpath_closed_0 | hard_or_ambiguous_even_strong_vlm | manual_review_gold_or_domain_reasoning |

## Next Experiment
- First target: ROI/multiview visual grounding on the recommended 10-sample set, without web.
- Compare: original image only vs multiview/ROI prompt vs stronger visual actor reference.
- Keep PRM mandatory, but log `candidate_gold_votes` and `PRM selected gold?` separately.
- For PRM rerank misses, test answer aggregation alternatives: max step score, last step score, min-step penalty, and pairwise option verification.
