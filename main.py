from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
import random
import re
import logging
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
import subprocess, tempfile, shutil
from fastapi.responses import FileResponse

# ── setup ──────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("cortiq")

app = FastAPI(title="Cortiq Backend API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "llama-3.3-70b-versatile"   # upgraded from 8b → 70b for quality


# ── groq client (cached) ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key)


# ── json extraction ────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """
    Tries three extraction strategies in order:
    1. Direct parse (model obeyed the instruction perfectly)
    2. Strip ```json ... ``` fences
    3. Regex scan for first {...} block
    """
    # 1. direct
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # 2. fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # 3. first {...} block
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        try:
            return json.loads(braces.group())
        except Exception:
            pass

    log.warning("extract_json failed on: %s", text[:200])
    return None


def llm(system: str, user: str, temperature: float = 0.4) -> Optional[dict]:
    """Single helper — calls Groq and returns parsed JSON or None."""
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        raw = resp.choices[0].message.content
        parsed = extract_json(raw)
        if not parsed:
            log.warning("LLM returned non-JSON: %s", raw[:300])
        return parsed
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return None


# ── data models ────────────────────────────────────────────────────────────────

class StartupInput(BaseModel):
    idea:               str   = Field(..., min_length=10)
    customer:           str   = Field(..., min_length=3)
    geography:          str   = ""
    tam:                float = Field(..., gt=0)
    competitors:        List[str] = []
    pricing:            float = 0
    cac:                float = 0
    monthly_burn:       float = 0
    current_revenue:    float = 0
    available_budget:   float = Field(..., gt=0)
    team_size:          int   = Field(..., gt=0)
    founder_experience: str   = "first_time"
    situation:          str   = ""

class DashboardOutput(BaseModel):
    health_score:       int
    risk_score:         float
    runway_months:      float
    market_health:      int
    competition_health: int
    execution_health:   int
    finance_health:     int
    growth_health:      int
    biggest_problem:    str
    improvements:       List[str]
    insight:            str


# ── scoring helpers ────────────────────────────────────────────────────────────

def calculate_runway(budget: float, burn: float) -> float:
    return 24.0 if burn <= 0 else round(budget / burn, 1)

def score_market(tam: float) -> int:
    if   tam > 10_000_000_000: return 95
    elif tam >  1_000_000_000: return 85
    elif tam >    100_000_000: return 70
    elif tam >     10_000_000: return 55
    else:                      return 30

def score_competition(competitors: List[str]) -> int:
    n = len([c for c in competitors if c.strip()])
    if   n == 0:  return 90
    elif n <= 2:  return 75
    elif n <= 5:  return 60
    elif n <= 10: return 50
    else:         return 40

def score_team(size: int, experience: str) -> int:
    score = 40
    if   size >= 6: score += 25
    elif size >= 3: score += 15
    elif size == 2: score += 10

    exp_bonus = {"repeat": 25, "experienced": 15, "technical": 10}
    score += exp_bonus.get(experience, 0)
    return min(score, 95)

def score_finance(runway: float, revenue: float, burn: float) -> int:
    score = 40
    if   runway > 18: score += 25
    elif runway > 12: score += 20
    elif runway >  6: score += 10
    if   revenue > burn: score += 20
    return min(score, 95)

def composite_health(m: int, c: int, t: int, f: int) -> int:
    raw    = m * 0.30 + c * 0.20 + t * 0.30 + f * 0.20
    jitter = random.randint(-3, 3)   # reduced from ±5 for stability
    return max(0, min(100, int(raw) + jitter))


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Cortiq backend running", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ── /dashboard/analyze ────────────────────────────────────────────────────────

_ANALYZE_FALLBACK = {
    "biggest_problem": "Strategic differentiation needs improvement.",
    "improvements": [
        "Sharpen product differentiation against competitors",
        "Extend runway by reducing non-essential burn",
        "Strengthen execution focus with weekly OKRs",
    ],
    "insight": "The startup shows moderate potential but requires stronger market positioning.",
}

@app.post("/dashboard/analyze", response_model=DashboardOutput)
def analyze(data: StartupInput):
    runway = calculate_runway(data.available_budget, data.monthly_burn)

    m = score_market(data.tam)
    c = score_competition(data.competitors)
    t = score_team(data.team_size, data.founder_experience)
    f = score_finance(runway, data.current_revenue, data.monthly_burn)

    health = composite_health(m, c, t, f)
    risk   = round(100 - health, 1)
    growth = int((m + t) / 2)

    system = (
        "You are a startup analyst. Respond with valid JSON only — "
        "no preamble, no markdown fences, no extra keys."
    )
    user = f"""
Startup: {data.idea}
Customer: {data.customer}  Geography: {data.geography}
TAM: ${data.tam:,.0f}  Competitors: {", ".join(data.competitors) or "none"}
Team: {data.team_size} people  Experience: {data.founder_experience}
Revenue: ${data.current_revenue:,.0f}/mo  Burn: ${data.monthly_burn:,.0f}/mo
Budget: ${data.available_budget:,.0f}  Runway: {runway} months
Situation: {data.situation or "N/A"}

Scores → Market:{m}  Competition:{c}  Team:{t}  Finance:{f}

Return EXACTLY this JSON structure:
{{
  "biggest_problem": "<one-sentence critical issue>",
  "improvements": ["<action 1>", "<action 2>", "<action 3>"],
  "insight": "<two-sentence strategic insight>"
}}
"""
    ai = llm(system, user) or _ANALYZE_FALLBACK

    return {
        "health_score":       health,
        "risk_score":         risk,
        "runway_months":      runway,
        "market_health":      m,
        "competition_health": c,
        "execution_health":   t,
        "finance_health":     f,
        "growth_health":      growth,
        "biggest_problem":    ai.get("biggest_problem", _ANALYZE_FALLBACK["biggest_problem"]),
        "improvements":       ai.get("improvements",    _ANALYZE_FALLBACK["improvements"]),
        "insight":            ai.get("insight",         _ANALYZE_FALLBACK["insight"]),
    }


# ── /investor-score ────────────────────────────────────────────────────────────

@app.post("/investor-score")
def investor_score(payload: dict):
    market      = payload.get("market_health",      0)
    team        = payload.get("execution_health",   0)
    finance     = payload.get("finance_health",     0)
    competition = payload.get("competition_health", 0)

    score = round(market * 0.35 + team * 0.25 + finance * 0.25 + competition * 0.15)

    if   score > 80: verdict = "Strong VC-scale startup"
    elif score > 65: verdict = "Promising early-stage opportunity"
    elif score > 50: verdict = "Needs stronger differentiation"
    else:            verdict = "High risk for investors"

    return {"investor_score": score, "verdict": verdict}


# ── /simulate ─────────────────────────────────────────────────────────────────

@app.post("/simulate")
def simulate(payload: dict):
    burn       = payload.get("monthly_burn", 0)
    budget     = payload.get("budget", 0)
    new_budget = payload.get("new_budget", budget)

    before = 24.0 if burn <= 0 else round(budget / burn, 1)
    after  = 24.0 if burn <= 0 else round(new_budget / burn, 1)

    return {"runway_before": before, "runway_after": after, "change": after - before}


# ── /market-research ──────────────────────────────────────────────────────────

_RESEARCH_FALLBACK = {"market_size": "Unknown", "growth_rate": "Unknown", "competitors": []}

@app.post("/market-research")
def market_research(payload: dict):
    idea = payload.get("idea", "")

    system = "You are a market research analyst. Return valid JSON only."
    user = f"""
Startup idea: {idea}

Return EXACTLY this JSON (use real-world estimates):
{{
  "market_size": "<e.g. $4.2B global market>",
  "growth_rate": "<e.g. 18% CAGR through 2028>",
  "competitors": ["<Competitor A>", "<Competitor B>", "<Competitor C>"]
}}
"""
    return llm(system, user, temperature=0.3) or _RESEARCH_FALLBACK


# ── /strategy ─────────────────────────────────────────────────────────────────

_STRATEGY_FALLBACK = {
    "strategy": [
        "Sharpen differentiation with a focused ICP and unique positioning",
        "Prioritise top-of-funnel experiments to validate CAC assumptions",
        "Reduce burn rate by deferring non-core hires until post-PMF",
    ]
}

@app.post("/strategy")
def strategy(payload: dict):
    idea    = payload.get("idea", "")
    metrics = payload.get("metrics", {})

    system = "You are a startup growth strategist. Return valid JSON only."
    user = f"""
Startup: {idea}

Health scores:
- Market: {metrics.get("market_health")}
- Competition: {metrics.get("competition_health")}
- Execution: {metrics.get("execution_health")}
- Finance: {metrics.get("finance_health")}
- Runway: {metrics.get("runway_months")} months

Identify the three highest-leverage strategic actions for the next 90 days.

Return EXACTLY:
{{
  "strategy": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"]
}}
"""
    return llm(system, user) or _STRATEGY_FALLBACK


# ── /startup-roadmap ──────────────────────────────────────────────────────────

_ROADMAP_FALLBACK = {
    "roadmap": [
        {"month": "Month 1",  "focus": "Validate Problem",    "tasks": ["Interview 20 potential customers", "Define core problem statement", "Map competitive landscape"]},
        {"month": "Month 2",  "focus": "Build MVP",           "tasks": ["Wireframe core user flows", "Ship v0.1 to 5 design partners", "Collect structured feedback"]},
        {"month": "Month 3",  "focus": "Iterate on Feedback", "tasks": ["Prioritise top 3 pain points", "Ship v0.2 with fixes", "Track weekly retention"]},
        {"month": "Month 4",  "focus": "First Revenue",       "tasks": ["Convert 2-3 design partners to paid", "Formalise pricing tiers", "Define success metrics"]},
        {"month": "Month 5",  "focus": "Growth Experiments",  "tasks": ["Run 3 acquisition channel tests", "Measure CAC per channel", "Double down on winner"]},
        {"month": "Month 6",  "focus": "Team & Process",      "tasks": ["Hire first key role", "Establish weekly OKR cadence", "Document core processes"]},
        {"month": "Month 7",  "focus": "Scale Distribution",  "tasks": ["Launch content / SEO strategy", "Build referral programme", "Expand to adjacent ICP"]},
        {"month": "Month 8",  "focus": "Product Depth",       "tasks": ["Ship highest-requested feature", "Reduce churn below 5%", "Launch customer success workflow"]},
        {"month": "Month 9",  "focus": "Fundraising Prep",    "tasks": ["Build pitch deck v1", "Identify 20 target investors", "Warm intros via network"]},
        {"month": "Month 10", "focus": "Investor Meetings",   "tasks": ["Run 10+ investor conversations", "Refine narrative from feedback", "Produce updated financials"]},
        {"month": "Month 11", "focus": "Close Round",         "tasks": ["Negotiate term sheet", "Complete due diligence", "Announce round"]},
        {"month": "Month 12", "focus": "Scale",               "tasks": ["Hire 3-5 roles with new capital", "Expand to new market segment", "Set 12-month OKRs"]},
    ]
}

@app.post("/startup-roadmap")
def startup_roadmap(payload: dict):
    system = (
        "You are a startup execution coach. Return valid JSON only — "
        "no markdown fences, no extra keys."
    )
    user = f"""
Startup analysis data:
{json.dumps(payload, indent=2)}

Create a realistic, specific 12-month execution roadmap tailored to this startup.
Each month must have a clear focus theme and 3 concrete tasks.

Return EXACTLY:
{{
  "roadmap": [
    {{"month": "Month 1", "focus": "<theme>", "tasks": ["<task1>", "<task2>", "<task3>"]}},
    ... (12 months total)
  ]
}}
"""
    result = llm(system, user, temperature=0.5)

    # validate shape — must have at least 6 months
    if result and isinstance(result.get("roadmap"), list) and len(result["roadmap"]) >= 6:
        return result

    return _ROADMAP_FALLBACK


# ── /improvement-guide ────────────────────────────────────────────────────────

_GUIDE_FALLBACK = {
    "sections": [
        {
            "title": "Sharpen Market Positioning",
            "explanation": "Without a clear, differentiated position you will struggle to convert awareness into paying customers.",
            "steps": [
                "Write a one-sentence value proposition targeting your specific ICP",
                "Identify the single pain point competitors ignore",
                "A/B test two landing page headlines measuring sign-up rate",
            ],
        },
        {
            "title": "Extend Your Runway",
            "explanation": "Every additional month of runway increases your options and reduces investor leverage.",
            "steps": [
                "Audit all monthly expenses and cut anything not tied to growth",
                "Negotiate annual contracts with vendors for 15-20% savings",
                "Model three burn scenarios: base, optimistic, and emergency",
            ],
        },
        {
            "title": "Accelerate Customer Validation",
            "explanation": "Speed of learning is your biggest competitive advantage at this stage.",
            "steps": [
                "Run 10 customer discovery calls this week",
                "Define a clear metric that proves problem-solution fit",
                "Ship a no-code prototype to your first five prospects",
            ],
        },
        {
            "title": "Strengthen Team Execution",
            "explanation": "Execution velocity separates funded startups from stalled ones.",
            "steps": [
                "Implement weekly OKRs with a Friday async review",
                "Remove the single biggest bottleneck in your shipping process",
                "Define ownership for each key metric across the team",
            ],
        },
    ]
}

@app.post("/improvement-guide")
def improvement_guide(payload: dict):
    system = (
        "You are a startup advisor. Return valid JSON only — "
        "no markdown, no preamble."
    )
    user = f"""
Startup performance data:
{json.dumps(payload, indent=2)}

Identify the 4 most impactful improvement areas for this startup.
For each area provide an explanation (2 sentences) and 3 specific, actionable steps.

Return EXACTLY:
{{
  "sections": [
    {{
      "title": "<area name>",
      "explanation": "<2-sentence explanation>",
      "steps": ["<step 1>", "<step 2>", "<step 3>"]
    }}
  ]
}}
"""
    result = llm(system, user, temperature=0.4)

    if result and isinstance(result.get("sections"), list) and len(result["sections"]) > 0:
        return result

    return _GUIDE_FALLBACK
    # ── ADD TO main.py ─────────────────────────────────────────────────────────────
# Drop both endpoints below into your existing main.py.
# No new dependencies — uses the existing llm() helper and Groq client.
# ───────────────────────────────────────────────────────────────────────────────


# ── /competitor-deep-dive ──────────────────────────────────────────────────────

_COMPETITOR_FALLBACK = {
    "competitors": [
        {
            "name": "Competitor A",
            "description": "Well-funded incumbent with broad feature set but poor UX.",
            "threat_level": "high",
            "threat_score": 72,
            "funding": "Series B — $18M",
            "founded": "2019",
            "pricing_model": "Per-seat SaaS",
            "scores": {
                "product": 75, "pricing": 60, "marketing": 80,
                "technology": 70, "support": 55, "brand": 78,
            },
            "strengths":  ["Strong brand recognition", "Large enterprise customer base", "Mature API ecosystem"],
            "weaknesses": ["Slow product iteration", "Expensive for SMBs", "Outdated UI/UX"],
            "your_edge":  ["10x faster onboarding", "AI-native architecture", "Transparent pricing"],
            "positioning_strategy": "Position against their enterprise complexity. Win on simplicity, speed, and modern AI-first design. Target their dissatisfied SMB customers first.",
        }
    ],
    "summary": {
        "market_position": "You are entering a market with established players but clear whitespace at the SMB/mid-market level.",
        "recommended_moat": "Build a data network effect early — the more customers use the product, the better the AI models become, creating a defensible moat.",
    }
}

# ── FIXED new_endpoints.py ────────────────────────────────────────────────────
# Fix 1: ValueError "Cannot specify ',' with 's'" in okr_generator
#         → form values come in as strings, not ints. Use safe int() conversion.
# Fix 2: extract_json failing on competitor-deep-dive
#         → LLM response was being truncated. Raise max_tokens to 3000.
# ───────────────────────────────────────────────────────────────────────────────


# ── STEP 1: replace your existing llm() helper with this version ───────────────
# The only change is adding the max_tokens parameter with a default of 1024.

def llm(system: str, user: str, temperature: float = 0.4, max_tokens: int = 1024):
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        raw = resp.choices[0].message.content
        parsed = extract_json(raw)
        if not parsed:
            log.warning("LLM returned non-JSON: %s", raw[:300])
        return parsed
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return None


# ── STEP 2: paste these two endpoints at the bottom of main.py ─────────────────


# ── /competitor-deep-dive ──────────────────────────────────────────────────────

_COMPETITOR_FALLBACK = {
    "competitors": [
        {
            "name": "Competitor A",
            "description": "Well-funded incumbent with broad feature set but poor UX.",
            "threat_level": "high",
            "threat_score": 72,
            "funding": "Series B - $18M",
            "founded": "2019",
            "pricing_model": "Per-seat SaaS",
            "scores": {
                "product": 75, "pricing": 60, "marketing": 80,
                "technology": 70, "support": 55, "brand": 78,
            },
            "strengths":  ["Strong brand recognition", "Large enterprise customer base", "Mature API ecosystem"],
            "weaknesses": ["Slow product iteration", "Expensive for SMBs", "Outdated UI/UX"],
            "your_edge":  ["10x faster onboarding", "AI-native architecture", "Transparent pricing"],
            "positioning_strategy": "Position against their enterprise complexity. Win on simplicity, speed, and modern AI-first design.",
        }
    ],
    "summary": {
        "market_position": "You are entering a market with established players but clear whitespace at the SMB/mid-market level.",
        "recommended_moat": "Build a data network effect early so the product improves with every customer added.",
    }
}

@app.post("/competitor-deep-dive")
def competitor_deep_dive(payload: dict):
    idea        = payload.get("idea", "")
    competitors = payload.get("competitors", "")
    metrics     = payload.get("metrics", {})

    if isinstance(competitors, list):
        comp_str = ", ".join([c for c in competitors if c.strip()])
    else:
        comp_str = str(competitors)

    if not comp_str:
        comp_str = "major players in this space"

    system = (
        "You are a competitive intelligence analyst. "
        "Return ONLY a single valid JSON object. "
        "No markdown fences, no explanation, no text before or after the JSON."
    )

    user = f"""Startup idea: {idea}
Our scores - Market:{metrics.get("market_health")} Execution:{metrics.get("execution_health")} Finance:{metrics.get("finance_health")}
Competitors: {comp_str}

Return ONLY this JSON structure:
{{
  "competitors": [
    {{
      "name": "string",
      "description": "2-sentence overview",
      "threat_level": "low or medium or high",
      "threat_score": 65,
      "funding": "e.g. Series B - $24M",
      "founded": "2018",
      "pricing_model": "e.g. Per-seat SaaS",
      "scores": {{
        "product": 70, "pricing": 60, "marketing": 75,
        "technology": 65, "support": 55, "brand": 70
      }},
      "strengths": ["strength 1", "strength 2", "strength 3"],
      "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
      "your_edge": ["your advantage 1", "advantage 2", "advantage 3"],
      "positioning_strategy": "2-sentence positioning strategy"
    }}
  ],
  "summary": {{
    "market_position": "2-sentence overall assessment",
    "recommended_moat": "2-sentence moat recommendation"
  }}
}}"""

    # max_tokens=3000 — profiles are large and were truncating at 1024
    result = llm(system, user, temperature=0.4, max_tokens=3000)

    if (
        result
        and isinstance(result.get("competitors"), list)
        and len(result["competitors"]) >= 1
        and result.get("summary")
    ):
        return result

    log.warning("competitor-deep-dive fell back to default")
    return _COMPETITOR_FALLBACK


# ── /okr-generator ─────────────────────────────────────────────────────────────

_OKR_FALLBACK = {
    "annual_goal": "Achieve product-market fit and reach $50k MRR by end of year",
    "quarters": [
        {
            "period": "Month 1-3",
            "theme": "Validate and Build",
            "completion": 0,
            "objectives": [
                {
                    "objective": "Achieve strong problem-solution fit with first customers",
                    "completion": 0,
                    "key_results": [
                        {"description": "Complete 30 customer discovery interviews",  "metric": "Interviews", "target": "30"},
                        {"description": "Reach NPS above 40 with beta users",        "metric": "NPS",        "target": "40+"},
                        {"description": "Ship MVP to 10 design partners",            "metric": "Partners",   "target": "10"},
                    ]
                },
                {
                    "objective": "Establish core team and execution rhythm",
                    "completion": 0,
                    "key_results": [
                        {"description": "Hire first 2 key roles (eng + GTM)",        "metric": "Hires",    "target": "2"},
                        {"description": "Implement weekly OKR review cadence",       "metric": "Cadence",  "target": "Weekly"},
                        {"description": "Define and track 5 core product metrics",   "metric": "Metrics",  "target": "5"},
                    ]
                }
            ]
        },
        {
            "period": "Month 4-6",
            "theme": "First Revenue",
            "completion": 0,
            "objectives": [
                {
                    "objective": "Generate first meaningful revenue",
                    "completion": 0,
                    "key_results": [
                        {"description": "Convert 5 design partners to paid",         "metric": "Customers", "target": "5"},
                        {"description": "Reach $5k MRR",                            "metric": "MRR",       "target": "$5k"},
                        {"description": "Achieve less than 5 percent monthly churn", "metric": "Churn",    "target": "< 5%"},
                    ]
                }
            ]
        },
        {
            "period": "Month 7-9",
            "theme": "Scale Distribution",
            "completion": 0,
            "objectives": [
                {
                    "objective": "Build repeatable customer acquisition",
                    "completion": 0,
                    "key_results": [
                        {"description": "Test 3 acquisition channels and double down on winner", "metric": "Channels", "target": "3"},
                        {"description": "Reach $20k MRR",                            "metric": "MRR",  "target": "$20k"},
                        {"description": "Reduce CAC by 30 percent from baseline",   "metric": "CAC",  "target": "-30%"},
                    ]
                }
            ]
        },
        {
            "period": "Month 10-12",
            "theme": "Fundraise and Accelerate",
            "completion": 0,
            "objectives": [
                {
                    "objective": "Close seed round and set up for Series A",
                    "completion": 0,
                    "key_results": [
                        {"description": "Run 20 or more investor meetings",          "metric": "Meetings", "target": "20+"},
                        {"description": "Reach $50k MRR before close",              "metric": "MRR",      "target": "$50k"},
                        {"description": "Close seed round at target valuation",      "metric": "Round",    "target": "Closed"},
                    ]
                }
            ]
        }
    ]
}

@app.post("/okr-generator")
def okr_generator(payload: dict):
    result   = payload.get("result", {})
    form     = payload.get("form", {})
    strategy = payload.get("strategy", [])
    roadmap  = payload.get("roadmap", [])
    horizon  = payload.get("horizon", "quarterly")

    # Safe conversion — form values arrive as strings from the frontend
    def to_int(val, default=0):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default

    revenue = to_int(form.get("current_revenue", 0))
    burn    = to_int(form.get("monthly_burn", 0))
    runway  = result.get("runway_months", 0)
    team    = to_int(form.get("team_size", 1))

    system = (
        "You are an OKR framework expert who has coached 200+ startups. "
        "Return ONLY a single valid JSON object — no markdown, no preamble."
    )

    user = f"""Startup: {form.get("idea", "")}
Customer: {form.get("customer", "")}  Team: {team} people
Revenue: ${revenue}/mo  Burn: ${burn}/mo  Runway: {runway} months
Founder experience: {form.get("founder_experience", "first_time")}
Situation: {form.get("situation", "")}
Health scores - Market:{result.get("market_health")} Execution:{result.get("execution_health")} Finance:{result.get("finance_health")} Growth:{result.get("growth_health")}
Strategic priorities: {", ".join(strategy) if strategy else "N/A"}
Roadmap themes: {", ".join([m.get("focus","") for m in (roadmap or [])[:6]]) or "N/A"}
Horizon: {horizon}

Return ONLY this JSON (4 quarters, 2-3 objectives each, 3 KRs each):
{{
  "annual_goal": "one ambitious annual north star goal",
  "quarters": [
    {{
      "period": "Month 1-3",
      "theme": "short theme in 4-6 words",
      "completion": 0,
      "objectives": [
        {{
          "objective": "objective statement",
          "completion": 0,
          "key_results": [
            {{"description": "specific measurable action", "metric": "metric name", "target": "target value"}}
          ]
        }}
      ]
    }}
  ]
}}"""

    result_data = llm(system, user, temperature=0.45, max_tokens=2500)

    if (
        result_data
        and isinstance(result_data.get("quarters"), list)
        and len(result_data["quarters"]) >= 4
        and result_data.get("annual_goal")
    ):
        return result_data

    log.warning("okr-generator fell back to default")
    return _OKR_FALLBACK

    # ── PASTE BOTH ENDPOINTS AT THE BOTTOM OF main.py ─────────────────────────────
# Also add these imports at the top of main.py if not already there:
#   import subprocess, tempfile, shutil
#   from fastapi.responses import FileResponse
#
# And run:  npm install -g pptxgenjs   (for the /pitch-deck/download endpoint)
# ───────────────────────────────────────────────────────────────────────────────


# ── /pitch-deck  (slide content generation) ───────────────────────────────────

_PITCH_FALLBACK = {
    "slides": [
        {"title": "Cover",          "tagline": "Building the future", "content": [], "speaker_note": "Introduce yourself and the company."},
        {"title": "Problem",        "tagline": "", "content": ["Large underserved pain point", "Existing solutions fall short", "Growing urgency"], "speaker_note": "Walk through the pain your customer feels today."},
        {"title": "Solution",       "tagline": "", "content": ["Purpose-built product", "10x better than status quo", "Defensible technology"], "speaker_note": "Show how your product solves the problem."},
        {"title": "Market Size",    "tagline": "", "content": ["$1B+ total addressable market", "18% CAGR through 2028", "Early market timing advantage"], "speaker_note": "Explain the market opportunity."},
        {"title": "Traction",       "tagline": "", "content": ["Early customer validation", "Strong week-over-week growth", "Key design partners signed"], "speaker_note": "Share your most impressive proof points."},
        {"title": "Business Model", "tagline": "", "content": ["SaaS subscription revenue", "Land and expand motion", "Negative churn target"], "speaker_note": "Explain how you make money."},
        {"title": "Competition",    "tagline": "", "content": ["Fragmented incumbent landscape", "No direct competitor at our layer", "Strong network effect moat"], "speaker_note": "Position yourself vs the landscape."},
        {"title": "Team",           "tagline": "", "content": ["Experienced founding team", "Deep domain expertise", "Prior exits and relevant background"], "speaker_note": "Explain why your team is uniquely positioned to win."},
        {"title": "Financials",     "tagline": "", "content": ["18+ months runway", "Clear path to profitability", "Capital efficient model"], "speaker_note": "Walk through your key financial metrics."},
        {"title": "The Ask",        "tagline": "", "content": ["Raising $X seed round", "Use of funds: product and GTM", "Target: Series A in 18 months"], "speaker_note": "State clearly what you are raising and why."},
    ]
}

@app.post("/pitch-deck")
def pitch_deck_content(payload: dict):
    result          = payload.get("result", {})
    form            = payload.get("form", {})
    strategy        = payload.get("strategy", [])
    market_research = payload.get("marketResearch", {})
    investor_score  = payload.get("investorScore", {})
    tone            = payload.get("tone", "investor")
    slides_count    = payload.get("slides_count", 10)

    # Safe int conversion for form string values
    def to_int(val, default=0):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default

    system = (
        "You are a world-class pitch deck writer who has helped companies raise "
        "over $500M. Return ONLY a single valid JSON object — "
        "no markdown fences, no explanation before or after."
    )

    user = f"""Startup: {form.get("idea", "")}
Customer: {form.get("customer", "")}  Geography: {form.get("geography", "")}
TAM: ${to_int(form.get("tam", 0)):,}  Team: {to_int(form.get("team_size", 1))} people
Revenue: ${to_int(form.get("current_revenue", 0))}/mo  Burn: ${to_int(form.get("monthly_burn", 0))}/mo
Budget: ${to_int(form.get("available_budget", 0))}  Runway: {result.get("runway_months", 0)} months
Competitors: {form.get("competitors", "")}
Situation: {form.get("situation", "")}
Health - Market:{result.get("market_health")} Execution:{result.get("execution_health")} Finance:{result.get("finance_health")}
Investor Score: {investor_score.get("investor_score") if investor_score else "N/A"}
Market Size: {market_research.get("market_size") if market_research else "N/A"}
Key insight: {result.get("insight", "")}
Strategy: {", ".join(strategy) if strategy else "N/A"}

Write a compelling {slides_count}-slide pitch deck for a {tone} audience.
Make it specific and data-rich — not generic.

Return ONLY this JSON:
{{
  "slides": [
    {{
      "title": "slide title",
      "tagline": "optional one-line hook (leave empty string if not needed)",
      "content": ["bullet 1", "bullet 2", "bullet 3"],
      "speaker_note": "what to say out loud for this slide"
    }}
  ]
}}"""

    result_data = llm(system, user, temperature=0.6, max_tokens=3000)

    if (
        result_data
        and isinstance(result_data.get("slides"), list)
        and len(result_data["slides"]) >= 5
    ):
        return result_data

    log.warning("pitch-deck fell back to default")
    return _PITCH_FALLBACK


# ── /pitch-deck/download  (generates .pptx via Node + pptxgenjs) ──────────────
# Requires: npm install -g pptxgenjs
# If Node is not available this endpoint returns 500 — the frontend falls back
# gracefully and the user can still use the slide content from /pitch-deck above.

import subprocess, tempfile, shutil
from fastapi.responses import FileResponse

DECK_SCRIPT = r"""
const pptxgen = require("pptxgenjs");
const payload = JSON.parse(process.argv[2]);
const slides  = payload.slides  || [];
const form    = payload.form    || {};
const outPath = payload.outPath;

const BG     = "050505";
const ACCENT = "C8FF00";
const CYAN   = "00E5FF";
const WHITE  = "FFFFFF";
const DIM    = "AAAAAA";
const BORDER = "222222";
const COLORS = [ACCENT,CYAN,"FFB800","FF6B6B","A855F7",ACCENT,CYAN,"FFB800","FF6B6B","A855F7",ACCENT,CYAN];

let pres    = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Cortiq AI";
pres.title  = form.idea || "Pitch Deck";

slides.forEach((slide, idx) => {
  const s     = pres.addSlide();
  const color = COLORS[idx % COLORS.length];
  const isCover = idx === 0;
  s.background = { color: BG };

  // left accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x:0, y:0, w:0.06, h:5.625,
    fill:{ color }, line:{ color, width:0 },
  });

  // slide number
  s.addText(String(idx+1).padStart(2,"0"), {
    x:9.2, y:0.15, w:0.55, h:0.28,
    fontFace:"Courier New", fontSize:7, color:"555555",
    align:"center", valign:"middle",
  });

  if (isCover) {
    s.addText("CORTIQ", {
      x:0.4, y:0.28, w:3, h:0.3,
      fontFace:"Courier New", fontSize:8, color:"444444", charSpacing:6,
    });
    s.addText(form.idea || slide.title || "Our Startup", {
      x:0.4, y:1.0, w:6.2, h:2.0,
      fontFace:"Arial Black", fontSize:36, color:WHITE, bold:true, valign:"top",
    });
    if (slide.tagline) {
      s.addText(slide.tagline, {
        x:0.4, y:3.15, w:6.0, h:0.5,
        fontFace:"Courier New", fontSize:13, color:color, italic:true,
      });
    }
    s.addShape(pres.shapes.LINE, {
      x:0.4, y:3.75, w:4, h:0, line:{ color, width:1.5 },
    });
    const meta = [form.customer, form.geography, new Date().getFullYear().toString()].filter(Boolean).join("  ·  ");
    s.addText(meta, {
      x:0.4, y:4.05, w:9, h:0.3,
      fontFace:"Courier New", fontSize:9, color:"555555",
    });
  } else {
    // dot + title
    s.addShape(pres.shapes.OVAL, {
      x:0.28, y:0.23, w:0.2, h:0.2,
      fill:{ color }, line:{ color, width:0 },
    });
    s.addText(slide.title || "", {
      x:0.6, y:0.14, w:8.5, h:0.58,
      fontFace:"Arial Black", fontSize:22, color:WHITE, bold:true, margin:0,
    });
    if (slide.tagline) {
      s.addText(slide.tagline, {
        x:0.6, y:0.76, w:8.5, h:0.36,
        fontFace:"Courier New", fontSize:11, color:color, italic:true,
      });
    }
    const bullets = (slide.content || []).slice(0, 5);
    const startY  = slide.tagline ? 1.28 : 1.05;
    const rowH    = Math.min(0.72, (4.1 - startY) / Math.max(bullets.length, 1));
    bullets.forEach((line, bi) => {
      const y = startY + bi * rowH;
      s.addShape(pres.shapes.RECTANGLE, {
        x:0.55, y:y+0.14, w:0.03, h:rowH*0.52,
        fill:{ color: color+"80" }, line:{ color: color+"80", width:0 },
      });
      s.addText(line, {
        x:0.72, y, w:9.0, h:rowH,
        fontFace:"Calibri", fontSize:14, color:"CCCCCC", valign:"middle",
      });
    });
    if (slide.speaker_note) s.addNotes(slide.speaker_note);
    s.addShape(pres.shapes.LINE, {
      x:0.4, y:5.28, w:9.2, h:0, line:{ color:BORDER, width:0.5 },
    });
    s.addText((slide.title || "").toUpperCase(), {
      x:0.4, y:5.34, w:9.2, h:0.22,
      fontFace:"Courier New", fontSize:6.5, color:"333333", charSpacing:3,
    });
  }
});

pres.writeFile({ fileName: outPath })
  .then(() => process.stdout.write("OK"))
  .catch(e => { process.stderr.write(e.message); process.exit(1); });
"""

@app.post("/pitch-deck/download")
def pitch_deck_download(payload: dict):
    tmp_dir = tempfile.mkdtemp()
    try:
        script_path = os.path.join(tmp_dir, "gen.js")
        out_path    = os.path.join(tmp_dir, "pitch.pptx")

        with open(script_path, "w") as f:
            f.write(DECK_SCRIPT)

        proc = subprocess.run(
            ["node", script_path, json.dumps({**payload, "outPath": out_path})],
            capture_output=True, text=True, timeout=30
        )

        if proc.returncode != 0 or not os.path.exists(out_path):
            log.error("pptxgenjs error: %s", proc.stderr)
            raise HTTPException(status_code=500, detail=f"PPTX generation failed: {proc.stderr}")

        return FileResponse(
            out_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename="cortiq-pitch-deck.pptx",
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # clean up temp dir after a short delay to allow FileResponse to stream
        import threading
        threading.Timer(10, shutil.rmtree, args=[tmp_dir], kwargs={"ignore_errors": True}).start()