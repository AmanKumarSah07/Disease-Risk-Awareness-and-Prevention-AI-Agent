from __future__ import annotations

from langchain_core.tools import tool

from tools.symptom_checker import check_symptoms

# Simplified high-prevalence hints (customize later with better epidemiology data).
HIGH_PREVALENCE: dict[str, list[str]] = {
    "Dengue Fever": ["bihar", "west bengal", "kolkata", "patna", "tropical", "monsoon", "india"],
    "Malaria": ["odisha", "chhattisgarh", "jharkhand", "assam", "tropical", "india"],
    "Tuberculosis": ["mumbai", "delhi", "kolkata", "urban", "dense", "india"],
}

# Pre-existing condition links (simplified)
CONDITION_RISKS: dict[str, list[str]] = {
    "obesity": ["Type 2 Diabetes", "Hypertension", "Heart Disease", "Chronic Kidney Disease"],
    "smoking": ["Heart Disease", "Hypertension", "Pneumonia", "Tuberculosis", "COVID-19"],
    "diabetes": ["Chronic Kidney Disease", "Heart Disease", "Pneumonia"],
    "hypertension": ["Heart Disease", "Chronic Kidney Disease"],
    "asthma": ["Pneumonia", "COVID-19", "Influenza (Flu)"],
}

CHRONIC_DISEASES = {
    "Type 2 Diabetes",
    "Hypertension",
    "Heart Disease",
    "Chronic Kidney Disease",
    "Hypothyroidism",
}

def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

@tool
def calculate_risk_score(symptoms: list[str], profile: dict) -> dict[str, float]:
    """
    Calculate risk scores (0.0-1.0) for diseases using:
      - symptom overlap score (0.5)
      - age modifier (0.2)
      - location modifier (0.2)
      - pre-existing condition modifier (0.1)

    Returns only diseases with final score > 0.2, sorted descending.
    """
    # Symptom overlap (0..1)
    overlap = check_symptoms.invoke({"symptoms": symptoms})  # tool invocation style
    if not overlap:
        return {}

    age = profile.get("age")
    try:
        age = int(age) if age is not None else None
    except Exception:
        age = None

    location = _norm(profile.get("location", ""))
    conditions = profile.get("conditions", []) or []
    conditions_norm = [_norm(c) for c in conditions]

    # Age modifier (0..0.2) - increases with age, but only applied to chronic diseases
    # baseline at 30, max at 80
    if age is None:
        age_mod = 0.0
    else:
        age_mod = min(max(age - 30, 0) / 50.0, 1.0) * 0.2  # (30->0) (80->0.2)

    results: dict[str, float] = {}
    for disease, sym_score in overlap.items():
        score = 0.0

        # 0.5 symptom contribution
        score += float(sym_score) * 0.5

        # 0.2 age contribution for chronic
        if disease in CHRONIC_DISEASES:
            score += age_mod

        # 0.2 location contribution if location matches any hint
        if disease in HIGH_PREVALENCE:
            if any(_norm(hint) in location for hint in HIGH_PREVALENCE[disease]):
                score += 0.2

        # 0.1 condition contribution if any linked condition present
        for cond in conditions_norm:
            if cond in CONDITION_RISKS and disease in CONDITION_RISKS[cond]:
                score += 0.1

        score = min(score, 1.0)
        score = round(score, 3)
        if score > 0.2:
            results[disease] = score

    return dict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))