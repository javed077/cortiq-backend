"""
genome.py — DNA tag extraction + Supabase persistence
Place this at: G:\\cortiq\\backend\\genome.py
"""

import logging
from typing import Optional

from supabase import Client

log = logging.getLogger("cortiq")

# ── DNA tag validation ─────────────────────────────────────────────────────────

_DNA_VALID = {
    "business_model":    {"saas", "marketplace", "infrastructure", "consumer", "api", "hardware", "services"},
    "defensibility":     {"network_effects", "data_moat", "switching_costs", "regulatory", "brand", "none_clear"},
    "market_timing":     {"ahead_of_market", "riding_wave", "mature_market", "behind_market"},
    "market_maturity":   {"emerging", "growing", "established", "declining"},
    "founder_archetype": {"domain_expert", "technical_outsider", "repeat_founder", "academic", "operator"},
    "primary_risk":      {"market_risk", "technical_risk", "regulatory_risk", "competition_risk", "execution_risk"},
    "go_to_market":      {"bottom_up_plg", "top_down_enterprise", "community", "marketplace_supply", "direct_sales"},
}

_DNA_SYSTEM = (
    "You are a startup classifier. "
    "Return ONLY a valid JSON object — no markdown, no preamble, no extra keys."
)


def extract_dna(idea: str, customer: str, llm_fn) -> Optional[dict]:
    """
    Extracts structured DNA tags from the startup idea using the existing llm() helper.
    Pass your llm function as llm_fn so genome.py stays decoupled from main.py.

    Usage in main.py:
        from genome import extract_dna
        dna = extract_dna(data.idea, data.customer, llm)
    """
    user = f"""Startup idea: {idea}
Target customer: {customer or "not specified"}

Classify this startup and return EXACTLY this JSON (pick one value per field):
{{
  "business_model":    "saas | marketplace | infrastructure | consumer | api | hardware | services",
  "defensibility":     "network_effects | data_moat | switching_costs | regulatory | brand | none_clear",
  "market_timing":     "ahead_of_market | riding_wave | mature_market | behind_market",
  "market_maturity":   "emerging | growing | established | declining",
  "founder_archetype": "domain_expert | technical_outsider | repeat_founder | academic | operator",
  "primary_risk":      "market_risk | technical_risk | regulatory_risk | competition_risk | execution_risk",
  "go_to_market":      "bottom_up_plg | top_down_enterprise | community | marketplace_supply | direct_sales"
}}"""

    result = llm_fn(_DNA_SYSTEM, user, temperature=0.2, max_tokens=200)
    if not result:
        log.warning("extract_dna: LLM returned nothing")
        return None

    # Validate — null out any tag that isn't in the allowed set
    for key, allowed in _DNA_VALID.items():
        val = result.get(key, "")
        # Strip pipes and whitespace in case model adds them
        val = val.strip().split("|")[0].strip() if isinstance(val, str) else ""
        if val in allowed:
            result[key] = val
        else:
            log.warning("extract_dna: invalid tag %s=%s", key, result.get(key))
            result[key] = None

    log.info("extract_dna: %s", result)
    return result


# ── Save idea to Supabase ──────────────────────────────────────────────────────

def save_idea(
    supabase: Client,
    user_id: str,
    data,           # StartupInput instance
    scores: dict,
    dna_tags: Optional[dict],
    ai: dict,
) -> Optional[str]:
    """
    Saves the full analysis to the ideas table.
    Returns the new idea_id UUID string, or None on failure.

    scores dict must have keys:
        health, market, execution, finance, growth, competition, risk, runway
    """
    try:
        record = supabase.table("ideas").insert({
            # Identity
            "user_id":              user_id,

            # Raw input
            "raw_input":            data.idea,
            "idea_text":            data.idea,
            "customer":             data.customer,
            "geography":            data.geography,

            # Financials
            "tam":                  data.tam,
            "team_size":            data.team_size,
            "founder_experience":   data.founder_experience,
            "monthly_burn":         data.monthly_burn,
            "current_revenue":      data.current_revenue,
            "available_budget":     data.available_budget,

            # Health scores
            "score_composite":      scores["health"],
            "score_market":         scores["market"],
            "score_execution":      scores["execution"],
            "score_finance":        scores["finance"],
            "score_growth":         scores["growth"],
            "score_competition":    scores["competition"],
            "risk_index":           scores["risk"],
            "runway_months":        scores["runway"],

            # AI outputs
            "biggest_problem":      ai.get("biggest_problem"),
            "insight":              ai.get("insight"),

            # Genome Layer 3
            "dna_tags":             dna_tags,

            # Outcome — empty until user/admin updates
            "outcome":              "unknown",
        }).execute()

        idea_id = record.data[0]["id"]
        log.info("save_idea: saved id=%s user=%s", idea_id, user_id)
        return idea_id

    except Exception as e:
        log.error("save_idea: failed — %s", e)
        return None