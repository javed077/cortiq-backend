from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
from dotenv import load_dotenv
from groq import Groq

# ================= SETUP =================
load_dotenv()
app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= GROQ =================
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing")
    return Groq(api_key=api_key)

# ================= MODELS =================
class StartupInput(BaseModel):
    mode: str
    strategy_mode: str
    idea: str
    customer: str
    geography: str
    tam: float
    competitors: List[str]
    pricing: float
    cac: float
    monthly_burn: float
    current_revenue: float
    available_budget: float
    team_size: int
    founder_experience: str
    situation: str

class DashboardOutput(BaseModel):
    health_score: int
    risk_score: float
    runway_months: float
    market_health: int
    competition_health: int
    execution_health: int
    finance_health: int
    growth_health: int
    biggest_problem: str
    improvements: List[str]
    insight: str

# ================= HEALTH =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= SCORING =================
def calculate_runway(budget, burn):
    if burn <= 0:
        return 24
    return round(budget / burn, 1)

def score_market(tam):
    if tam > 1_000_000_000: return 90
    if tam > 100_000_000: return 75
    if tam > 10_000_000: return 60
    return 40

def score_competition(competitors):
    count = len(competitors)
    if count == 0: return 85
    if count < 3: return 70
    if count < 6: return 55
    return 35

def score_team(size, experience):
    score = 50
    if size >= 5: score += 15
    elif size >= 3: score += 10
    if experience == "repeat": score += 20
    elif experience == "experienced": score += 10
    return min(score, 95)

def score_finance(runway, revenue, burn):
    score = 50
    if runway > 12: score += 20
    elif runway > 6: score += 10
    if revenue > burn: score += 20
    return min(score, 95)

def calculate_risk(m, c, t, f):
    weighted = m*0.25 + c*0.20 + t*0.25 + f*0.30
    return round(100 - weighted, 1)

# ================= ANALYZE =================
@app.post("/dashboard/analyze", response_model=DashboardOutput)
def analyze(data: StartupInput):

    runway = calculate_runway(data.available_budget, data.monthly_burn)
    m = score_market(data.tam)
    c = score_competition(data.competitors)
    t = score_team(data.team_size, data.founder_experience)
    f = score_finance(runway, data.current_revenue, data.monthly_burn)

    risk = calculate_risk(m, c, t, f)
    health = int(max(0, min(100, 100 - risk)))
    growth = int((m + t) / 2)

    # ---------- SAFE DEFAULT ----------
    ai_result = {
        "biggest_problem": "Structural positioning and capital efficiency need refinement.",
        "improvements": [
            "Clarify market differentiation strategy",
            "Improve capital efficiency and extend runway",
            "Strengthen execution capability through focused hiring"
        ],
        "insight": "The startup demonstrates moderate structural potential but requires improved strategic focus, capital efficiency, and operational discipline before aggressive scaling."
    }

    try:
        client = get_client()

        prompt = f"""
You are a senior startup strategist.

Return STRICT valid JSON only.
No markdown.
No explanation.
No ellipsis.
No placeholders.

Minimum 3 actionable improvements.
Insight must be minimum 40 words.

Format:

{{
"biggest_problem": "specific structural weakness",
"improvements": ["action 1", "action 2", "action 3"],
"insight": "deep strategic explanation"
}}

Startup Metrics:
Market Score: {m}
Competition Score: {c}
Team Score: {t}
Finance Score: {f}
Risk Score: {risk}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a strict JSON API."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()

        start = content.find("{")
        end = content.rfind("}") + 1

        if start != -1 and end != -1:
            parsed = json.loads(content[start:end])

            if (
                isinstance(parsed, dict)
                and parsed.get("biggest_problem")
                and parsed["biggest_problem"] != "..."
                and isinstance(parsed.get("improvements"), list)
                and len(parsed["improvements"]) >= 3
                and len(parsed.get("insight", "")) > 40
            ):
                ai_result = parsed

    except Exception as e:
        print("AI ERROR:", str(e))

    return {
        "health_score": health,
        "risk_score": risk,
        "runway_months": runway,
        "market_health": m,
        "competition_health": c,
        "execution_health": t,
        "finance_health": f,
        "growth_health": growth,
        "biggest_problem": ai_result["biggest_problem"],
        "improvements": ai_result["improvements"],
        "insight": ai_result["insight"]
    }

# ================= INTERACTIVE AI COACH =================
@app.post("/coach/chat")
def coach_chat(payload: dict):

    message = payload.get("message", "")
    metrics = payload.get("metrics", {})

    safe_reply = (
        "Focus on strengthening differentiation, improving capital efficiency, "
        "and reinforcing execution discipline to increase structural strength."
    )

    try:
        client = get_client()

        prompt = f"""
You are Cortiq AI Coach.
Act like a strategic startup advisor.

Startup Metrics:
{json.dumps(metrics)}

User Question:
{message}

Respond:
- Clear
- Direct
- Strategic
- Actionable
- No markdown
- No fluff
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a high-level startup strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        reply = response.choices[0].message.content.strip()

        if len(reply) < 20:
            return {"reply": safe_reply}

        return {"reply": reply}

    except Exception as e:
        print("COACH ERROR:", str(e))
        return {"reply": safe_reply}