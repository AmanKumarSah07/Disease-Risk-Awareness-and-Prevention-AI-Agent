from __future__ import annotations

from langchain_core.tools import tool

# Hardcoded mapping of common diseases to typical symptoms (non-exhaustive).
# This is for *risk awareness*, not diagnosis.
DISEASE_SYMPTOMS: dict[str, list[str]] = {
    "Type 2 Diabetes": ["increased thirst", "frequent urination", "increased hunger", "fatigue", "blurred vision", "slow healing wounds"],
    "Hypertension": ["headache", "dizziness", "nosebleeds", "shortness of breath", "chest pain"],
    "Tuberculosis": ["persistent cough", "coughing up blood", "chest pain", "fever", "night sweats", "weight loss", "fatigue"],
    "Dengue Fever": ["high fever", "severe headache", "pain behind eyes", "joint pain", "muscle pain", "rash", "nausea"],
    "Malaria": ["fever", "chills", "sweats", "headache", "nausea", "vomiting", "body aches", "fatigue"],
    "COVID-19": ["fever", "cough", "sore throat", "shortness of breath", "fatigue", "loss of taste", "loss of smell", "body aches"],
    "Typhoid": ["prolonged fever", "fatigue", "headache", "abdominal pain", "diarrhea", "constipation", "loss of appetite"],
    "Anemia": ["fatigue", "weakness", "pale skin", "shortness of breath", "dizziness", "cold hands", "cold feet"],
    "Asthma": ["shortness of breath", "chest tightness", "wheezing", "coughing", "trouble sleeping due to breathing"],
    "Chronic Kidney Disease": ["fatigue", "swelling in legs", "changes in urine output", "nausea", "vomiting", "loss of appetite", "sleep problems"],
    "Heart Disease": ["chest pain", "chest tightness", "shortness of breath", "fatigue", "pain in arm", "jaw pain", "nausea"],
    "Hypothyroidism": ["fatigue", "weight gain", "constipation", "dry skin", "cold intolerance", "depressed mood", "slow heart rate"],
    "Hepatitis B": ["fatigue", "abdominal pain", "dark urine", "nausea", "vomiting", "loss of appetite", "yellowing of skin", "joint pain"],
    "Cholera": ["watery diarrhea", "vomiting", "dehydration", "muscle cramps", "thirst"],
    "Pneumonia": ["cough with phlegm", "fever", "chills", "shortness of breath", "chest pain", "fatigue"],
    "Influenza (Flu)": ["fever", "chills", "cough", "sore throat", "runny nose", "body aches", "fatigue", "headache"],
    "Migraine": ["severe headache", "nausea", "vomiting", "sensitivity to light", "sensitivity to sound", "visual aura"],
    "Gastroenteritis": ["diarrhea", "nausea", "vomiting", "abdominal cramps", "fever", "dehydration"],
    "UTI (Urinary Tract Infection)": ["burning urination", "frequent urination", "urgent urination", "lower abdominal pain", "cloudy urine"],
    "Depression": ["low mood", "loss of interest", "fatigue", "sleep changes", "appetite changes", "difficulty concentrating"],
    "Anxiety": ["restlessness", "rapid heartbeat", "sweating", "trembling", "difficulty concentrating", "sleep problems"],
}

def _normalize(symptom: str) -> str:
    return " ".join(symptom.strip().lower().split())

@tool
def check_symptoms(symptoms: list[str]) -> dict[str, float]:
    """
    Map symptoms to possibly associated diseases using a hardcoded dictionary.
    Returns {disease_name: symptom_overlap_score}, where score is the fraction
    of the disease's typical symptoms that match the input.

    NOTE: This does NOT diagnose. It's only for risk awareness.
    """
    normalized = [_normalize(s) for s in symptoms if s and s.strip()]
    if not normalized:
        return {}

    results: dict[str, float] = {}
    for disease, typical in DISEASE_SYMPTOMS.items():
        typical_norm = [_normalize(t) for t in typical]

        # simple containment / fuzzy-ish match: if symptom is substring of typical or vice versa
        overlap = 0
        for ts in typical_norm:
            if any((ts in s) or (s in ts) for s in normalized):
                overlap += 1

        if overlap > 0:
            results[disease] = round(overlap / max(len(typical_norm), 1), 3)

    return dict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))