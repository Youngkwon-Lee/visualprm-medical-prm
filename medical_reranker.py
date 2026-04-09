"""
Medical-domain reranking and adaptive context selection utilities.
"""

from __future__ import annotations

import re


PATHOLOGY_TERMS = {
    "nucleus",
    "cytoplasm",
    "stain",
    "fibrosis",
    "necrosis",
    "gland",
    "epithelium",
    "cell",
    "tissue",
    "histology",
}

RADIOLOGY_TERMS = {
    "lung",
    "pleural",
    "opacity",
    "lobe",
    "fracture",
    "mass",
    "transverse",
    "axial",
    "lesion",
    "herniation",
}

ANATOMY_TERMS = {
    "brain", "cerebellum", "brainstem", "ventricle", "kidney", "liver", "colon", "rib",
    "lung", "pleural", "spine", "skull", "pelvis", "heart", "aorta", "bladder",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", (text or "").lower()))


def rerank_hits(
    *,
    question: str,
    options: list[str],
    case_type: str,
    modality: str,
    question_type: str,
    hits: list,
) -> list[dict]:
    question_tokens = _tokenize(question)
    option_tokens = _tokenize(" ".join(options))
    query_anatomy = question_tokens & ANATOMY_TERMS
    domain_terms = set()
    if "path" in (case_type or "").lower() or "h&e" in (modality or "").lower():
        domain_terms = PATHOLOGY_TERMS
    elif "radio" in (case_type or "").lower() or any(token in (modality or "").lower() for token in ("ct", "mri", "x-ray")):
        domain_terms = RADIOLOGY_TERMS

    reranked = []
    for idx, hit in enumerate(hits):
        hit_tokens = _tokenize(" ".join([hit.question, hit.case_type, hit.modality, " ".join(hit.options)]))
        lexical_overlap = len(question_tokens & hit_tokens)
        option_overlap = len(option_tokens & hit_tokens)
        domain_overlap = len(domain_terms & hit_tokens) if domain_terms else 0

        hit_case_type = (hit.case_type or "").lower()
        hit_modality = (hit.modality or "").lower()
        modality_lower = (modality or "").lower()
        case_type_lower = (case_type or "").lower()

        modality_bonus = 0.5 if modality_lower and hit_modality and hit_modality == modality_lower else 0.0
        case_type_bonus = 0.6 if case_type_lower and hit_case_type and hit_case_type == case_type_lower else 0.0

        # Penalize obvious domain mismatch (e.g., chest/rib hit for brain question)
        mismatch_penalty = 0.0
        if domain_terms and domain_overlap == 0:
            mismatch_penalty += 0.45
        if case_type_lower and hit_case_type and hit_case_type != case_type_lower:
            mismatch_penalty += 0.25

        hit_anatomy = hit_tokens & ANATOMY_TERMS
        anatomy_bonus = 0.0
        if query_anatomy and hit_anatomy:
            if query_anatomy & hit_anatomy:
                anatomy_bonus += 0.35
            else:
                mismatch_penalty += 0.65

        question_type_bonus = 0.25 if question_type == "yes_no" and len(hit.options) == 2 else 0.0
        if question_type == "pattern_interpretation" and any(token in hit.question.lower() for token in ("uptake", "pattern", "tracer")):
            question_type_bonus += 0.35
        base_score = float(getattr(hit, "score", 0.0))

        # yes/no questions are often lexically similar; rely more on domain/modality than pure token overlap
        if question_type == "yes_no":
            lexical_weight = 0.20
            option_weight = 0.10
            domain_weight = 0.50
        else:
            lexical_weight = 0.35
            option_weight = 0.20
            domain_weight = 0.35

        rerank_score = (
            (1.35 * base_score)
            + (lexical_weight * lexical_overlap)
            + (option_weight * option_overlap)
            + (domain_weight * domain_overlap)
            + modality_bonus
            + case_type_bonus
            + anatomy_bonus
            + question_type_bonus
            - mismatch_penalty
            - (0.03 * idx)
        )

        reranked.append(
            {
                "hit": hit,
                "rerank_score": round(rerank_score, 3),
                "reasons": {
                    "base_score": round(base_score, 3),
                    "lexical_overlap": lexical_overlap,
                    "option_overlap": option_overlap,
                    "domain_overlap": domain_overlap,
                    "modality_bonus": modality_bonus,
                    "case_type_bonus": case_type_bonus,
                    "anatomy_bonus": anatomy_bonus,
                    "mismatch_penalty": mismatch_penalty,
                    "question_type_bonus": question_type_bonus,
                },
            }
        )

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked


def select_adaptive_hits(reranked_hits: list[dict], max_hits: int = 3) -> list[dict]:
    if not reranked_hits:
        return []

    top_score = reranked_hits[0]["rerank_score"]
    if len(reranked_hits) == 1:
        return reranked_hits[:1]

    selected = [reranked_hits[0]]
    for item in reranked_hits[1:max_hits]:
        if item["rerank_score"] >= top_score * 0.72:
            selected.append(item)

    return selected[:max_hits]
