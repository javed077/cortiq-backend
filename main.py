from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json, random, re, logging, subprocess, tempfile, shutil, threading
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

# ── Genome imports ─────────────────────────────────────────────────────────────
from db import get_supabase, get_user_id
from genome import extract_dna, save_idea

# ── setup ──────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("cortiq")

app = FastAPI(title="Cortiq Backend API", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "llama-3.3-70b-versatile"


# ── groq client ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key)


# ── json extraction ────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        try:
            return json.loads(braces.group())
        except Exception:
            pass
    log.warning("extract_json failed on: %s", text[:200])
    return None


def llm(system: str, user: str, temperature: float = 0.4, max_tokens: int = 1024) -> Optional[dict]:
    """Calls Groq and returns parsed JSON or None."""
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30,
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


def llm_text(system: str, user: str, temperature: float = 0.6, max_tokens: int = 800) -> str:
    """Calls Groq and returns raw text (for non-JSON responses)."""
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=40,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.error("LLM text call failed: %s", exc)
        return ""


def llm_chat(system: str, messages: list, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """Multi-turn chat — accepts full message history."""
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.error("LLM chat call failed: %s", exc)
        return ""


# ── safe int helper ────────────────────────────────────────────────────────────

def to_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


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
    idea_id:            Optional[str] = None  # ← new: returned when user is logged in


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
    jitter = random.randint(-3, 3)
    return max(0, min(100, int(raw) + jitter))


# ══════════════════════════════════════════════════════════════════════════════
# CORE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Cortiq backend running", "version": "4.1.0"}

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
def analyze(
    data: StartupInput,
    user_id: Optional[str] = Depends(get_user_id),   # ← reads JWT, None if not logged in
):
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

Return EXACTLY this JSON:
{{
  "biggest_problem": "<one-sentence critical issue>",
  "improvements": ["<action 1>", "<action 2>", "<action 3>"],
  "insight": "<two-sentence strategic insight>"
}}
"""
    ai = llm(system, user) or _ANALYZE_FALLBACK

    # ── Genome persistence ─────────────────────────────────────────────────────
    idea_id = None
    if user_id:
        try:
            dna_tags = extract_dna(data.idea, data.customer, llm)
            scores = {
                "health":      health,
                "market":      m,
                "execution":   t,
                "finance":     f,
                "growth":      growth,
                "competition": c,
                "risk":        risk,
                "runway":      runway,
            }
            idea_id = save_idea(get_supabase(), user_id, data, scores, dna_tags, ai)
        except Exception as e:
            # Never fail the request because of persistence — just log it
            log.error("Genome persistence failed: %s", e)
    else:
        log.info("analyze: no auth token — skipping Genome save")
    # ──────────────────────────────────────────────────────────────────────────

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
        "idea_id":            idea_id,
    }


# ── /ideas/history ─────────────────────────────────────────────────────────────

@app.get("/ideas/history")
def ideas_history(
    user_id: Optional[str] = Depends(get_user_id),
    limit: int = 10,
):
    """
    Returns the logged-in user's past analyses, newest first.
    Useful for showing a history page or benchmarking over time.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        result = (
            get_supabase()
            .table("ideas")
            .select(
                "id, created_at, idea_text, score_composite, score_market, "
                "score_execution, score_finance, score_growth, score_competition, "
                "risk_index, runway_months, biggest_problem, insight, dna_tags, outcome"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"ideas": result.data, "count": len(result.data)}
    except Exception as e:
        log.error("ideas/history failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not fetch history")


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

@app.post("/market-research")
def market_research(payload: dict):
    idea   = payload.get("idea", "")
    system = "You are a market research analyst. Return valid JSON only."
    user   = f"""Startup idea: {idea}
Return EXACTLY:
{{
  "market_size": "<e.g. $4.2B global market>",
  "growth_rate": "<e.g. 18% CAGR through 2028>",
  "competitors": ["<Competitor A>", "<Competitor B>", "<Competitor C>"]
}}"""
    return llm(system, user, temperature=0.3) or {"market_size": "Unknown", "growth_rate": "Unknown", "competitors": []}


# ── /strategy ─────────────────────────────────────────────────────────────────

@app.post("/strategy")
def strategy(payload: dict):
    idea    = payload.get("idea", "")
    metrics = payload.get("metrics", {})
    system  = "You are a startup growth strategist. Return valid JSON only."
    user    = f"""Startup: {idea}
Health scores — Market:{metrics.get("market_health")} Competition:{metrics.get("competition_health")} Execution:{metrics.get("execution_health")} Finance:{metrics.get("finance_health")} Runway:{metrics.get("runway_months")} months
Identify the three highest-leverage strategic actions for the next 90 days.
Return EXACTLY: {{"strategy": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"]}}"""
    return llm(system, user) or {"strategy": ["Sharpen differentiation with a focused ICP","Prioritise top-of-funnel experiments to validate CAC","Reduce burn by deferring non-core hires until post-PMF"]}


# ── /startup-roadmap ──────────────────────────────────────────────────────────

_ROADMAP_FALLBACK = {"roadmap": [
    {"month": "Month 1",  "focus": "Validate Problem",    "tasks": ["Interview 20 potential customers","Define core problem statement","Map competitive landscape"]},
    {"month": "Month 2",  "focus": "Build MVP",           "tasks": ["Wireframe core user flows","Ship v0.1 to 5 design partners","Collect structured feedback"]},
    {"month": "Month 3",  "focus": "Iterate on Feedback", "tasks": ["Prioritise top 3 pain points","Ship v0.2 with fixes","Track weekly retention"]},
    {"month": "Month 4",  "focus": "First Revenue",       "tasks": ["Convert 2-3 design partners to paid","Formalise pricing tiers","Define success metrics"]},
    {"month": "Month 5",  "focus": "Growth Experiments",  "tasks": ["Run 3 acquisition channel tests","Measure CAC per channel","Double down on winner"]},
    {"month": "Month 6",  "focus": "Team & Process",      "tasks": ["Hire first key role","Establish weekly OKR cadence","Document core processes"]},
    {"month": "Month 7",  "focus": "Scale Distribution",  "tasks": ["Launch content / SEO strategy","Build referral programme","Expand to adjacent ICP"]},
    {"month": "Month 8",  "focus": "Product Depth",       "tasks": ["Ship highest-requested feature","Reduce churn below 5%","Launch customer success workflow"]},
    {"month": "Month 9",  "focus": "Fundraising Prep",    "tasks": ["Build pitch deck v1","Identify 20 target investors","Warm intros via network"]},
    {"month": "Month 10", "focus": "Investor Meetings",   "tasks": ["Run 10+ investor conversations","Refine narrative from feedback","Produce updated financials"]},
    {"month": "Month 11", "focus": "Close Round",         "tasks": ["Negotiate term sheet","Complete due diligence","Announce round"]},
    {"month": "Month 12", "focus": "Scale",               "tasks": ["Hire 3-5 roles with new capital","Expand to new market segment","Set 12-month OKRs"]},
]}

@app.post("/startup-roadmap")
def startup_roadmap(payload: dict):
    system = "You are a startup execution coach. Return valid JSON only — no markdown fences, no extra keys."
    user   = f"""Startup analysis data:
{json.dumps(payload, indent=2)}
Create a realistic, specific 12-month execution roadmap tailored to this startup. Each month must have a clear focus theme and 3 concrete tasks.
Return EXACTLY:
{{"roadmap": [{{"month": "Month 1", "focus": "<theme>", "tasks": ["<task1>", "<task2>", "<task3>"]}}]}} (12 months total)"""
    result = llm(system, user, temperature=0.5)
    if result and isinstance(result.get("roadmap"), list) and len(result["roadmap"]) >= 6:
        return result
    return _ROADMAP_FALLBACK


# ── /improvement-guide ────────────────────────────────────────────────────────

_GUIDE_FALLBACK = {"sections": [
    {"title": "Sharpen Market Positioning", "explanation": "Without a clear, differentiated position you will struggle to convert awareness into paying customers.", "steps": ["Write a one-sentence value proposition targeting your specific ICP","Identify the single pain point competitors ignore","A/B test two landing page headlines measuring sign-up rate"]},
    {"title": "Extend Your Runway",         "explanation": "Every additional month of runway increases your options and reduces investor leverage.",                   "steps": ["Audit all monthly expenses and cut anything not tied to growth","Negotiate annual contracts with vendors for 15-20% savings","Model three burn scenarios: base, optimistic, and emergency"]},
    {"title": "Accelerate Customer Validation", "explanation": "Speed of learning is your biggest competitive advantage at this stage.",                              "steps": ["Run 10 customer discovery calls this week","Define a clear metric that proves problem-solution fit","Ship a no-code prototype to your first five prospects"]},
    {"title": "Strengthen Team Execution",  "explanation": "Execution velocity separates funded startups from stalled ones.",                                         "steps": ["Implement weekly OKRs with a Friday async review","Remove the single biggest bottleneck in your shipping process","Define ownership for each key metric across the team"]},
]}

@app.post("/improvement-guide")
def improvement_guide(payload: dict):
    system = "You are a startup advisor. Return valid JSON only — no markdown, no preamble."
    user   = f"""Startup performance data:
{json.dumps(payload, indent=2)}
Identify the 4 most impactful improvement areas. For each provide an explanation (2 sentences) and 3 specific actionable steps.
Return EXACTLY:
{{"sections": [{{"title": "<area>", "explanation": "<2 sentences>", "steps": ["<step 1>", "<step 2>", "<step 3>"]}}]}}"""
    result = llm(system, user, temperature=0.4)
    if result and isinstance(result.get("sections"), list) and len(result["sections"]) > 0:
        return result
    return _GUIDE_FALLBACK


# ── /competitor-deep-dive ──────────────────────────────────────────────────────

_COMPETITOR_FALLBACK = {"competitors": [{"name": "Competitor A","description": "Well-funded incumbent with broad feature set but poor UX.","threat_level": "high","threat_score": 72,"funding": "Series B — $18M","founded": "2019","pricing_model": "Per-seat SaaS","scores": {"product": 75,"pricing": 60,"marketing": 80,"technology": 70,"support": 55,"brand": 78},"strengths": ["Strong brand recognition","Large enterprise customer base","Mature API ecosystem"],"weaknesses": ["Slow product iteration","Expensive for SMBs","Outdated UI/UX"],"your_edge": ["10x faster onboarding","AI-native architecture","Transparent pricing"],"positioning_strategy": "Position against their enterprise complexity. Win on simplicity, speed, and modern AI-first design."}],"summary": {"market_position": "You are entering a market with established players but clear whitespace at the SMB/mid-market level.","recommended_moat": "Build a data network effect early so the product improves with every customer added."}}

@app.post("/competitor-deep-dive")
def competitor_deep_dive(payload: dict):
    idea        = payload.get("idea", "")
    competitors = payload.get("competitors", "")
    metrics     = payload.get("metrics", {})
    comp_str    = ", ".join([c for c in competitors if c.strip()]) if isinstance(competitors, list) else str(competitors)
    if not comp_str:
        comp_str = "major players in this space"

    system = "You are a competitive intelligence analyst. Return ONLY a single valid JSON object. No markdown fences, no text before or after."
    user   = f"""Startup idea: {idea}
Our scores — Market:{metrics.get("market_health")} Execution:{metrics.get("execution_health")} Finance:{metrics.get("finance_health")}
Competitors: {comp_str}
Return ONLY:
{{"competitors": [{{"name": "string","description": "2-sentence overview","threat_level": "low|medium|high","threat_score": 65,"funding": "e.g. Series B — $24M","founded": "2018","pricing_model": "e.g. Per-seat SaaS","scores": {{"product": 70,"pricing": 60,"marketing": 75,"technology": 65,"support": 55,"brand": 70}},"strengths": ["s1","s2","s3"],"weaknesses": ["w1","w2","w3"],"your_edge": ["e1","e2","e3"],"positioning_strategy": "2-sentence strategy"}}],"summary": {{"market_position": "2-sentence assessment","recommended_moat": "2-sentence moat recommendation"}}}}"""

    result = llm(system, user, temperature=0.4, max_tokens=3000)
    if result and isinstance(result.get("competitors"), list) and len(result["competitors"]) >= 1 and result.get("summary"):
        return result
    log.warning("competitor-deep-dive used fallback")
    return _COMPETITOR_FALLBACK


# ── /okr-generator ─────────────────────────────────────────────────────────────

_OKR_FALLBACK = {"annual_goal": "Achieve product-market fit and reach $50k MRR by end of year","quarters": [{"period": "Month 1-3","theme": "Validate and Build","completion": 0,"objectives": [{"objective": "Achieve strong problem-solution fit with first customers","completion": 0,"key_results": [{"description": "Complete 30 customer discovery interviews","metric": "Interviews","target": "30"},{"description": "Reach NPS above 40 with beta users","metric": "NPS","target": "40+"},{"description": "Ship MVP to 10 design partners","metric": "Partners","target": "10"}]},{"objective": "Establish core team and execution rhythm","completion": 0,"key_results": [{"description": "Hire first 2 key roles (eng + GTM)","metric": "Hires","target": "2"},{"description": "Implement weekly OKR review cadence","metric": "Cadence","target": "Weekly"},{"description": "Define and track 5 core product metrics","metric": "Metrics","target": "5"}]}]},{"period": "Month 4-6","theme": "First Revenue","completion": 0,"objectives": [{"objective": "Generate first meaningful revenue","completion": 0,"key_results": [{"description": "Convert 5 design partners to paid","metric": "Customers","target": "5"},{"description": "Reach $5k MRR","metric": "MRR","target": "$5k"},{"description": "Achieve less than 5% monthly churn","metric": "Churn","target": "< 5%"}]}]},{"period": "Month 7-9","theme": "Scale Distribution","completion": 0,"objectives": [{"objective": "Build repeatable customer acquisition","completion": 0,"key_results": [{"description": "Test 3 acquisition channels and double down on winner","metric": "Channels","target": "3"},{"description": "Reach $20k MRR","metric": "MRR","target": "$20k"},{"description": "Reduce CAC by 30% from baseline","metric": "CAC","target": "-30%"}]}]},{"period": "Month 10-12","theme": "Fundraise and Accelerate","completion": 0,"objectives": [{"objective": "Close seed round and set up for Series A","completion": 0,"key_results": [{"description": "Run 20+ investor meetings","metric": "Meetings","target": "20+"},{"description": "Reach $50k MRR before close","metric": "MRR","target": "$50k"},{"description": "Close seed round at target valuation","metric": "Round","target": "Closed"}]}]}]}

@app.post("/okr-generator")
def okr_generator(payload: dict):
    result   = payload.get("result",   {})
    form     = payload.get("form",     {})
    strategy = payload.get("strategy", [])
    roadmap  = payload.get("roadmap",  [])
    horizon  = payload.get("horizon",  "quarterly")

    revenue = to_int(form.get("current_revenue", 0))
    burn    = to_int(form.get("monthly_burn", 0))
    team    = to_int(form.get("team_size", 1))

    system = "You are an OKR framework expert. Return ONLY a single valid JSON object — no markdown, no preamble."
    user   = f"""Startup: {form.get("idea","")}  Customer: {form.get("customer","")}  Team: {team} people
Revenue: ${revenue}/mo  Burn: ${burn}/mo  Runway: {result.get("runway_months",0)} months
Health scores — Market:{result.get("market_health")} Execution:{result.get("execution_health")} Finance:{result.get("finance_health")} Growth:{result.get("growth_health")}
Strategic priorities: {", ".join(strategy) if strategy else "N/A"}
Horizon: {horizon}
Return ONLY (4 quarters, 2-3 objectives each, 3 KRs each):
{{"annual_goal": "north star goal","quarters": [{{"period": "Month 1-3","theme": "4-6 words","completion": 0,"objectives": [{{"objective": "statement","completion": 0,"key_results": [{{"description": "action","metric": "name","target": "value"}}]}}]}}]}}"""

    result_data = llm(system, user, temperature=0.45, max_tokens=2500)
    if result_data and isinstance(result_data.get("quarters"), list) and len(result_data["quarters"]) >= 4 and result_data.get("annual_goal"):
        return result_data
    log.warning("okr-generator used fallback")
    return _OKR_FALLBACK


# ── /pitch-deck ────────────────────────────────────────────────────────────────

_PITCH_FALLBACK = {"slides": [
    {"title": "Cover",          "tagline": "Building the future", "content": [],                                                                           "speaker_note": "Introduce yourself and the company."},
    {"title": "Problem",        "tagline": "",                    "content": ["Large underserved pain point","Existing solutions fall short","Growing urgency"], "speaker_note": "Walk through the pain your customer feels today."},
    {"title": "Solution",       "tagline": "",                    "content": ["Purpose-built product","10x better than status quo","Defensible technology"],    "speaker_note": "Show how your product solves the problem."},
    {"title": "Market Size",    "tagline": "",                    "content": ["$1B+ total addressable market","18% CAGR through 2028","Early timing advantage"], "speaker_note": "Explain the market opportunity."},
    {"title": "Traction",       "tagline": "",                    "content": ["Early customer validation","Strong week-over-week growth","Key design partners"], "speaker_note": "Share your most impressive proof points."},
    {"title": "Business Model", "tagline": "",                    "content": ["SaaS subscription revenue","Land and expand motion","Negative churn target"],     "speaker_note": "Explain how you make money."},
    {"title": "Competition",    "tagline": "",                    "content": ["Fragmented incumbent landscape","No direct competitor at our layer","Network moat"],"speaker_note": "Position yourself vs the landscape."},
    {"title": "Team",           "tagline": "",                    "content": ["Experienced founding team","Deep domain expertise","Prior exits and background"],  "speaker_note": "Explain why your team is uniquely positioned to win."},
    {"title": "Financials",     "tagline": "",                    "content": ["18+ months runway","Clear path to profitability","Capital efficient model"],       "speaker_note": "Walk through your key financial metrics."},
    {"title": "The Ask",        "tagline": "",                    "content": ["Raising $X seed round","Use of funds: product and GTM","Series A in 18 months"],  "speaker_note": "State clearly what you are raising and why."},
]}

@app.post("/pitch-deck")
def pitch_deck_content(payload: dict):
    result          = payload.get("result", {})
    form            = payload.get("form",   {})
    strategy        = payload.get("strategy", [])
    market_research = payload.get("marketResearch", {})
    investor_score  = payload.get("investorScore", {})
    tone            = payload.get("tone", "investor")
    slides_count    = payload.get("slides_count", 10)

    system = "You are a world-class pitch deck writer. Return ONLY a single valid JSON object — no markdown fences, no explanation."
    user   = f"""Startup: {form.get("idea","")}  Customer: {form.get("customer","")}  Geography: {form.get("geography","")}
TAM: ${to_int(form.get("tam",0)):,}  Team: {to_int(form.get("team_size",1))} people
Revenue: ${to_int(form.get("current_revenue",0))}/mo  Burn: ${to_int(form.get("monthly_burn",0))}/mo  Runway: {result.get("runway_months",0)} months
Health — Market:{result.get("market_health")} Execution:{result.get("execution_health")} Finance:{result.get("finance_health")}
Investor Score: {investor_score.get("investor_score") if investor_score else "N/A"}
Market Size: {market_research.get("market_size") if market_research else "N/A"}
Key insight: {result.get("insight","")}  Strategy: {", ".join(strategy) if strategy else "N/A"}
Write a compelling {slides_count}-slide pitch deck for a {tone} audience. Be specific and data-rich.
Return ONLY: {{"slides": [{{"title": "slide title","tagline": "optional hook or empty string","content": ["bullet 1","bullet 2","bullet 3"],"speaker_note": "what to say"}}]}}"""

    result_data = llm(system, user, temperature=0.6, max_tokens=3000)
    if result_data and isinstance(result_data.get("slides"), list) and len(result_data["slides"]) >= 5:
        return result_data
    log.warning("pitch-deck used fallback")
    return _PITCH_FALLBACK


# ── /pitch-deck/download ──────────────────────────────────────────────────────

DECK_SCRIPT = r"""
const pptxgen = require("pptxgenjs");
const payload = JSON.parse(process.argv[2]);
const slides  = payload.slides  || [];
const form    = payload.form    || {};
const outPath = payload.outPath;
const BG="050505",ACCENT="C8FF00",CYAN="00E5FF",WHITE="FFFFFF",BORDER="222222";
const COLORS=[ACCENT,CYAN,"FFB800","FF6B6B","A855F7",ACCENT,CYAN,"FFB800","FF6B6B","A855F7",ACCENT,CYAN];
let pres=new pptxgen(); pres.layout="LAYOUT_16x9"; pres.author="Cortiq AI"; pres.title=form.idea||"Pitch Deck";
slides.forEach((slide,idx)=>{
  const s=pres.addSlide(); const color=COLORS[idx%COLORS.length]; const isCover=idx===0;
  s.background={color:BG};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:0.06,h:5.625,fill:{color},line:{color,width:0}});
  s.addText(String(idx+1).padStart(2,"0"),{x:9.2,y:0.15,w:0.55,h:0.28,fontFace:"Courier New",fontSize:7,color:"555555",align:"center",valign:"middle"});
  if(isCover){
    s.addText("CORTIQ",{x:0.4,y:0.28,w:3,h:0.3,fontFace:"Courier New",fontSize:8,color:"444444",charSpacing:6});
    s.addText(form.idea||slide.title||"Our Startup",{x:0.4,y:1.0,w:6.2,h:2.0,fontFace:"Arial Black",fontSize:36,color:WHITE,bold:true,valign:"top"});
    if(slide.tagline)s.addText(slide.tagline,{x:0.4,y:3.15,w:6.0,h:0.5,fontFace:"Courier New",fontSize:13,color:color,italic:true});
    s.addShape(pres.shapes.LINE,{x:0.4,y:3.75,w:4,h:0,line:{color,width:1.5}});
    const meta=[form.customer,form.geography,new Date().getFullYear().toString()].filter(Boolean).join("  ·  ");
    s.addText(meta,{x:0.4,y:4.05,w:9,h:0.3,fontFace:"Courier New",fontSize:9,color:"555555"});
  }else{
    s.addShape(pres.shapes.OVAL,{x:0.28,y:0.23,w:0.2,h:0.2,fill:{color},line:{color,width:0}});
    s.addText(slide.title||"",{x:0.6,y:0.14,w:8.5,h:0.58,fontFace:"Arial Black",fontSize:22,color:WHITE,bold:true,margin:0});
    if(slide.tagline)s.addText(slide.tagline,{x:0.6,y:0.76,w:8.5,h:0.36,fontFace:"Courier New",fontSize:11,color:color,italic:true});
    const bullets=(slide.content||[]).slice(0,5); const startY=slide.tagline?1.28:1.05;
    const rowH=Math.min(0.72,(4.1-startY)/Math.max(bullets.length,1));
    bullets.forEach((line,bi)=>{
      const y=startY+bi*rowH;
      s.addShape(pres.shapes.RECTANGLE,{x:0.55,y:y+0.14,w:0.03,h:rowH*0.52,fill:{color:color+"80"},line:{color:color+"80",width:0}});
      s.addText(line,{x:0.72,y,w:9.0,h:rowH,fontFace:"Calibri",fontSize:14,color:"CCCCCC",valign:"middle"});
    });
    if(slide.speaker_note)s.addNotes(slide.speaker_note);
    s.addShape(pres.shapes.LINE,{x:0.4,y:5.28,w:9.2,h:0,line:{color:BORDER,width:0.5}});
    s.addText((slide.title||"").toUpperCase(),{x:0.4,y:5.34,w:9.2,h:0.22,fontFace:"Courier New",fontSize:6.5,color:"333333",charSpacing:3});
  }
});
pres.writeFile({fileName:outPath}).then(()=>process.stdout.write("OK")).catch(e=>{process.stderr.write(e.message);process.exit(1);});
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
            capture_output=True, text=True, timeout=30,
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
        threading.Timer(10, shutil.rmtree, args=[tmp_dir], kwargs={"ignore_errors": True}).start()


# ── /export-pdf ───────────────────────────────────────────────────────────────

@app.post("/export-pdf")
def export_pdf(payload: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed. Run: pip install reportlab")

    result        = payload.get("result", {})
    form          = payload.get("form",   {})
    strategy      = payload.get("strategy", [])
    market        = payload.get("marketResearch", {})
    investor      = payload.get("investorScore", {})
    tmp_dir       = tempfile.mkdtemp()
    out_path      = os.path.join(tmp_dir, "cortiq-report.pdf")

    try:
        doc    = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        GREEN  = colors.HexColor("#C8FF00")
        DARK   = colors.HexColor("#050505")
        story  = []

        h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=24, textColor=DARK, spaceAfter=6)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, textColor=DARK, spaceAfter=4)
        body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, spaceAfter=4, leading=14)

        story.append(Paragraph(f"Cortiq Analysis Report", h1))
        story.append(Paragraph(f"Startup: {form.get('idea','—')}  |  Customer: {form.get('customer','—')}", body))
        story.append(Spacer(1, 8*mm))

        story.append(Paragraph("Key Metrics", h2))
        metrics_data = [
            ["Health Score", str(result.get("health_score","—")), "Risk Index", str(result.get("risk_score","—"))],
            ["Runway",       f"{result.get('runway_months','—')} months", "Investor Score", str(investor.get("investor_score","—")) if investor else "—"],
        ]
        t = Table(metrics_data, colWidths=[45*mm, 35*mm, 45*mm, 35*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F5F5F5")),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 6*mm))

        story.append(Paragraph("Key Insight", h2))
        story.append(Paragraph(result.get("insight", "—"), body))
        story.append(Spacer(1, 4*mm))

        story.append(Paragraph("Critical Risk", h2))
        story.append(Paragraph(result.get("biggest_problem", "—"), body))
        story.append(Spacer(1, 4*mm))

        if result.get("improvements"):
            story.append(Paragraph("Recommended Improvements", h2))
            for i, imp in enumerate(result["improvements"], 1):
                story.append(Paragraph(f"{i}. {imp}", body))
            story.append(Spacer(1, 4*mm))

        if strategy:
            story.append(Paragraph("Strategic Recommendations", h2))
            for s in strategy:
                story.append(Paragraph(f"• {s}", body))
            story.append(Spacer(1, 4*mm))

        if market:
            story.append(Paragraph("Market Research", h2))
            story.append(Paragraph(f"Market Size: {market.get('market_size','—')}  |  Growth Rate: {market.get('growth_rate','—')}", body))

        doc.build(story)
        return FileResponse(out_path, media_type="application/pdf", filename="cortiq-analysis.pdf")
    except Exception as e:
        log.error("export-pdf failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        threading.Timer(10, shutil.rmtree, args=[tmp_dir], kwargs={"ignore_errors": True}).start()


# ── /weekly-digest ────────────────────────────────────────────────────────────

_DIGEST_FALLBACK = {"digest": "WEEKLY STARTUP DIGEST\n\nNo data yet — run your first analysis to start tracking.\n\nKeep coming back each week to build your streak.\n\nSee you next week. Stay consistent."}

@app.post("/weekly-digest")
def weekly_digest(payload: dict):
    history      = payload.get("history", [])
    streak       = payload.get("streak", {})
    achievements = payload.get("achievements", [])

    if not history:
        return _DIGEST_FALLBACK

    latest = history[-1] if history else {}
    prev   = history[-2] if len(history) >= 2 else {}
    delta  = latest.get("health_score", 0) - prev.get("health_score", 0) if prev else None
    delta_str = f"+{delta}" if delta and delta > 0 else str(delta) if delta is not None else "first entry"
    unlocked = [a for a in achievements if a.get("unlocked")]
    locked   = [a for a in achievements if not a.get("unlocked")]
    dims = {"market": latest.get("market_health",0), "execution": latest.get("execution_health",0), "finance": latest.get("finance_health",0), "growth": latest.get("growth_health",0), "competition": latest.get("competition_health",0)}
    weakest = min(dims, key=dims.get)

    system = "You are a startup coach writing a weekly digest. Be direct, specific, honest. Use plain text with section headers — no markdown symbols."
    user   = f"""Startup: {latest.get("idea","the startup")}
Health score: {latest.get("health_score",0)} ({delta_str} vs last week)
Streak: {streak.get("current",0)} weeks (longest: {streak.get("longest",0)})
Achievements: {len(unlocked)}/{len(achievements)} unlocked
Weakest dimension: {weakest} ({dims[weakest]})
Score history (last 5): {[h.get("health_score") for h in history[-5:]]}
Write a concise weekly digest: 1) SCORE THIS WEEK 2) BIGGEST WIN 3) THIS WEEK'S FOCUS 4) STREAK UPDATE 5) NEXT MILESTONE
2-3 sentences each. Direct startup coach tone."""

    text = llm_text(system, user)
    return {"digest": text} if text else _DIGEST_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# GENOME ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_GENOME_PROMPTS = {
    "strategy": """You are a brutally honest McKinsey-tier strategy consultant. Analyze this startup idea using Porter's Five Forces. Be sharp and punchy — no filler.

Format your response exactly like this:

FORCES ASSESSMENT
• Competitive Rivalry: [LOW/MED/HIGH/VERY HIGH] — [1-line reason]
• Buyer Power: [rating] — [reason]
• Supplier Power: [rating] — [reason]
• Threat of New Entrants: [rating] — [reason]
• Threat of Substitutes: [rating] — [reason]

TOP FAILURE RISKS
1. [risk] — [why it kills the business]
2. [risk] — [why]
3. [risk] — [why]

STRATEGIC DEFENSES
1. [defense]
2. [defense]
3. [defense]

VERDICT: [1 brutal honest sentence]""",

    "market": """You are a rigorous market analyst. For this startup, calculate TAM, SAM, SOM using bottom-up logic. Show real math.

Format exactly:

TAM: [population] × [price/month × 12] = $[result]
SAM: [addressable segment description] = $[result]
SOM (3yr): [realistic capture] = $[result] ARR / [N] users

KEY ASSUMPTIONS
• [assumption 1]
• [assumption 2]
• [assumption 3]

WEAK ASSUMPTIONS (flag what is speculative)
• [what could be wrong]

YC READINESS: [1 sentence — is this market sizing compelling for YC?]""",

    "financial": """You are a startup CFO. Build a 3-year P&L projection with three scenarios.

Format exactly:

YEAR 1
• Conservative: $Xk revenue / $Xk costs / $X profit (loss)
• Base: $Xk / $Xk / $X
• Aggressive: $Xk / $Xk / $X

YEAR 2
• Conservative: ...
• Base: ...
• Aggressive: ...

YEAR 3
• Conservative: ...
• Base: ...
• Aggressive: ...

UNIT ECONOMICS
• CAC: $X
• LTV: $X
• LTV:CAC: X.X
• Payback period: X months

MOST SENSITIVE VARIABLES
1. [variable] — [why it swings outcomes most]
2. [variable]
3. [variable]""",

    "customer": """You ARE the target customer for this product. Respond with brutal, honest objections. Do not be polite.

Format exactly:

TOP OBJECTIONS (ranked by deal-breaking importance)
1. [objection] — [why this almost stops you buying]
2. [objection] — [why]
3. [objection] — [why]
4. [objection] — [why]
5. [objection] — [why]

TRUST BARRIERS
• [what would make you trust this enough to pay]
• [what proof you need]

PRICING REACTION
[Your gut reaction to paying for this — be honest]

POSITIONING THAT WOULD ACTUALLY LAND
1. "[message that would convert you]"
2. "[message]"
3. "[message]" """,

    "competitive": """You are a competitive intelligence analyst. Analyze the competitive landscape for this startup.

Format exactly:

CLOSEST COMPETITORS
1. [Company] — Model: [X] | Moat: [X] | Key weakness: [X]
2. [Company] — Model: [X] | Moat: [X] | Key weakness: [X]
3. [Company] — Model: [X] | Moat: [X] | Key weakness: [X]

MARKET ATTACK PLAN (steal 20% share in 18 months)
Step 1: [specific action]
Step 2: [specific action]
Step 3: [specific action]

DIFFERENTIATION WINDOW
[What window of opportunity exists and how long it stays open]

MOAT ASSESSMENT: [None / Weak / Moderate / Strong] — [1-line reason]""",
}

_GENOME_SUMMARY_FALLBACK = {
    "idea_score": "—",
    "biggest_risk": "See strategy module",
    "market_size": "See market module",
    "top_objection": "See customer module",
    "competitor_gap": "See competitor module",
    "recommendation": "Review the module outputs above for detailed recommendations.",
}


class GenomeModuleRequest(BaseModel):
    idea:     str = Field(..., min_length=5)
    customer: str = ""
    module:   str = Field(..., pattern="^(strategy|market|financial|customer|competitive)$")

class GenomeAnalyzeRequest(BaseModel):
    idea:     str = Field(..., min_length=5)
    customer: str = ""
    modules:  List[str] = ["strategy", "market", "financial", "customer", "competitive"]

class GenomeSummaryRequest(BaseModel):
    idea:        str = ""
    strategy:    str = ""
    market:      str = ""
    financial:   str = ""
    customer:    str = ""
    competitive: str = ""


@app.post("/genome/module")
def genome_module(req: GenomeModuleRequest):
    prompt = _GENOME_PROMPTS.get(req.module)
    if not prompt:
        raise HTTPException(status_code=400, detail=f"Unknown module: {req.module}")
    user_context = f"Startup idea: {req.idea}\nTarget customer: {req.customer or 'not specified'}"
    content = llm_text(prompt, user_context, temperature=0.55, max_tokens=900)
    if not content:
        return {"module": req.module, "content": "Analysis unavailable — please retry.", "ok": False}
    return {"module": req.module, "content": content, "ok": True}


@app.post("/genome/analyze")
def genome_analyze(req: GenomeAnalyzeRequest):
    valid_modules = [m for m in req.modules if m in _GENOME_PROMPTS]
    if not valid_modules:
        raise HTTPException(status_code=400, detail="No valid modules specified")
    user_context = f"Startup idea: {req.idea}\nTarget customer: {req.customer or 'not specified'}"
    results: dict[str, str] = {}
    for module_id in valid_modules:
        prompt  = _GENOME_PROMPTS[module_id]
        content = llm_text(prompt, user_context, temperature=0.55, max_tokens=900)
        results[module_id] = content or "Analysis unavailable — please retry."
        log.info("genome/analyze: completed module=%s idea=%.40s", module_id, req.idea)
    return {"idea": req.idea, "modules": results, "ok": True}


@app.post("/genome/summary")
def genome_summary(req: GenomeSummaryRequest):
    parts = []
    if req.strategy:    parts.append(f"[STRATEGY]\n{req.strategy}")
    if req.market:      parts.append(f"[MARKET]\n{req.market}")
    if req.financial:   parts.append(f"[FINANCIAL]\n{req.financial}")
    if req.customer:    parts.append(f"[CUSTOMER]\n{req.customer}")
    if req.competitive: parts.append(f"[COMPETITIVE]\n{req.competitive}")
    if not parts:
        return _GENOME_SUMMARY_FALLBACK
    combined = "\n\n---\n\n".join(parts)
    system = (
        "You are the Cortiq Genome Engine. Synthesize the provided intelligence module outputs "
        "into a crisp founder summary. Return ONLY valid JSON — no markdown fences, no preamble. "
        "Use this exact shape:\n"
        '{\n'
        '  "idea_score": "7.4 / 10",\n'
        '  "biggest_risk": "one concise phrase",\n'
        '  "market_size": "$X.XB TAM",\n'
        '  "top_objection": "one concise phrase",\n'
        '  "competitor_gap": "one concise phrase describing the opening to exploit",\n'
        '  "recommendation": "2-3 sentence actionable recommendation for this specific startup"\n'
        '}'
    )
    user = f"Startup: {req.idea or 'not specified'}\n\nModule outputs to synthesize:\n\n{combined}"
    result = llm(system, user, temperature=0.4, max_tokens=600)
    if result and result.get("idea_score") and result.get("recommendation"):
        log.info("genome/summary: synthesized ok idea=%.40s", req.idea)
        return result
    log.warning("genome/summary used fallback")
    return _GENOME_SUMMARY_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# MENTOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class MentorTaskRequest(BaseModel):
    idea:               str       = ""
    customer:           str       = ""
    market_health:      int       = 50
    execution_health:   int       = 50
    finance_health:     int       = 50
    growth_health:      int       = 50
    competition_health: int       = 50
    biggest_problem:    str       = ""
    recent_areas:       List[str] = []
    tasks_done:         int       = 0
    tasks_total:        int       = 0

class MentorChatMessage(BaseModel):
    role:    str
    content: str

class MentorChatRequest(BaseModel):
    idea:            str                    = ""
    health_score:    int                    = 0
    biggest_problem: str                    = ""
    recent_tasks:    List[str]              = []
    messages:        List[MentorChatMessage] = []

class MentorWeeklyRequest(BaseModel):
    tasks_done:     int       = 0
    tasks_total:    int       = 0
    done_titles:    List[str] = []
    skipped_titles: List[str] = []
    reflections:    List[str] = []
    score_change:   Optional[int] = None


_ALL_AREAS = ["market", "execution", "finance", "growth", "competition"]

_TASK_FALLBACK = {
    "area":  "execution",
    "title": "Map your riskiest assumption and design a fast test for it",
    "why":   "Every stalled startup has one wrong assumption at its core — finding yours today beats six months of wasted build.",
    "steps": [
        "Write down the 3 biggest assumptions your business depends on",
        "Pick the one that would kill the company if wrong",
        "Design the cheapest, fastest experiment to validate or kill it this week",
    ],
}

_WEEKLY_FALLBACK = {
    "verdict": "steady",
    "summary": "You completed some tasks this week but consistency is still building. The habit of daily execution is more important than any single task.",
    "nudge":   "Show up every day this week — completion rate matters less than the streak.",
}


@app.post("/mentor/daily-task")
def mentor_daily_task(req: MentorTaskRequest):
    area_scores = {
        "market":      req.market_health,
        "execution":   req.execution_health,
        "finance":     req.finance_health,
        "growth":      req.growth_health,
        "competition": req.competition_health,
    }

    def priority(area: str) -> float:
        recency_penalty = req.recent_areas.count(area) * 20
        return area_scores[area] - recency_penalty

    focus_area = min(_ALL_AREAS, key=priority)

    system = (
        "You are a tough but caring startup mentor. "
        "Give ONE specific, high-impact daily task that can be completed in 1-3 hours. "
        "Be direct and concrete — no vague advice. "
        "Return valid JSON only, no markdown."
    )
    user = f"""Startup: {req.idea or "early-stage startup"}
Target customer: {req.customer or "unknown"}
Focus area today: {focus_area}
Health scores — Market:{req.market_health} Execution:{req.execution_health} Finance:{req.finance_health} Growth:{req.growth_health} Competition:{req.competition_health}
Biggest problem: {req.biggest_problem or "not specified"}
Task completion so far: {req.tasks_done} of {req.tasks_total} done

Return EXACTLY:
{{
  "area": "{focus_area}",
  "title": "<specific task, max 12 words, starts with a verb>",
  "why": "<one direct sentence: why THIS task matters RIGHT NOW>",
  "steps": ["<concrete step 1>", "<concrete step 2>", "<concrete step 3>"]
}}"""

    result = llm(system, user, temperature=0.6)
    if result and result.get("title") and result.get("steps"):
        result["area"] = focus_area
        return result

    _TASK_FALLBACK["area"] = focus_area
    return _TASK_FALLBACK


@app.post("/mentor/weekly-report")
def mentor_weekly_report(req: MentorWeeklyRequest):
    completion_pct = round((req.tasks_done / req.tasks_total) * 100) if req.tasks_total > 0 else 0

    if   completion_pct >= 70: prior = "improving"
    elif completion_pct >= 40: prior = "steady"
    else:                      prior = "slipping"

    score_line = ""
    if req.score_change is not None:
        direction  = "up" if req.score_change > 0 else "down" if req.score_change < 0 else "unchanged"
        score_line = f"Health score moved {direction} by {abs(req.score_change)} points this week."

    system = (
        "You are a startup mentor giving a weekly review. "
        "Be honest and direct — no sugarcoating. "
        "If slipping, say so clearly. If improving, acknowledge it genuinely. "
        "Return valid JSON only."
    )
    user = f"""Weekly performance:
- Tasks completed: {req.tasks_done} of {req.tasks_total} ({completion_pct}%)
- Done: {', '.join(req.done_titles) or 'none'}
- Skipped: {', '.join(req.skipped_titles) or 'none'}
- Founder reflections: {'; '.join(req.reflections) or 'none provided'}
- {score_line}
- Numbers suggest: {prior}

Return EXACTLY:
{{
  "verdict": "improving|steady|slipping",
  "summary": "<2 sentences: what happened and what it means>",
  "nudge": "<1 direct sentence from mentor — honest, specific, no fluff>"
}}"""

    result = llm(system, user, temperature=0.5)
    if result and result.get("verdict") in ("improving", "steady", "slipping"):
        return result
    return _WEEKLY_FALLBACK


@app.post("/mentor/chat")
def mentor_chat(req: MentorChatRequest):
    system = (
        f"You are a tough but caring startup mentor advising: '{req.idea or 'an early-stage startup'}'. "
        f"Their current health score is {req.health_score}/100. "
        f"Biggest problem: {req.biggest_problem or 'not specified'}. "
        f"Recent tasks: {', '.join(req.recent_tasks) if req.recent_tasks else 'none yet'}. "
        "Give short, direct, practical advice — max 3 sentences unless more depth is explicitly requested. "
        "Use real examples. Call out BS if you see it. No fluff, no preamble."
    )

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply    = llm_chat(system, messages, temperature=0.7, max_tokens=512)
    return {"reply": reply or "Connection error — please try again."}