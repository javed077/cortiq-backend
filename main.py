from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
import hashlib
import time

from dotenv import load_dotenv
from groq import Groq

# ======================
# SETUP
# ======================
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

app = FastAPI()

# ======================
# CACHE
# ======================
analysis_cache = {}
cache_timestamps = {}
CACHE_TTL = 30  # seconds

# ======================
# CORS
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# INPUT MODEL
# ======================
class StartupInput(BaseModel):
    mode: str
    strategy_mode: str = ""
    idea: str = ""
    customer: str
    pricing: str
    team_size: int
    budget: str
    launch_timeline: str = ""
    situation: str = ""


# ======================
# OUTPUT MODEL
# ======================
class DashboardOutput(BaseModel):
    health_score: int
    market_health: int
    execution_health: int
    finance_health: int
    growth_health: int
    biggest_problem: str
    improvements: List[str]
    insight: str


@app.get("/")
def root():
    return {"message": "Cortiq Health API running 🚀"}


# ======================
# CACHE KEY
# ======================
def generate_cache_key(data: StartupInput):
    raw = f"""
{data.mode}
{data.strategy_mode}
{data.idea}
{data.customer}
{data.pricing}
{data.team_size}
{data.budget}
{data.launch_timeline}
{data.situation}
"""
    return hashlib.md5(raw.encode()).hexdigest()


# ======================
# DASHBOARD ANALYSIS
# ======================
@app.post("/dashboard/analyze", response_model=DashboardOutput)
def analyze_dashboard(data: StartupInput):

    print("INPUT RECEIVED:", data)

    # ---------- CACHE ----------
    cache_key = generate_cache_key(data)

    if cache_key in analysis_cache:
        age = time.time() - cache_timestamps[cache_key]
        if age < CACHE_TTL:
            print("⚡ Returning cached result")
            return analysis_cache[cache_key]

    # ==================================================
    # 🔥 DETERMINISTIC SCORING (REAL FIX)
    # ==================================================

    health_score = 70

    # TEAM SIZE IMPACT
    if data.team_size <= 2:
        health_score -= 15
    elif data.team_size <= 5:
        health_score -= 5
    elif data.team_size > 50:
        health_score += 5

    # BUDGET IMPACT
    budget = data.budget.lower()
    if "low" in budget or "0" in budget:
        health_score -= 15
    elif "10 lakh" in budget or "high" in budget:
        health_score += 5

    # IDEA QUALITY
    if len(data.idea) < 10:
        health_score -= 10
    else:
        health_score += 5

    # CURRENT SITUATION
    situation = data.situation.lower()
    if "building" in situation:
        health_score += 3
    if "idea" in situation:
        health_score -= 5

    health_score = max(0, min(100, health_score))

    # SUB SCORES
    market_health = max(0, min(100, health_score - 5))
    execution_health = max(0, min(100, health_score + 3))
    finance_health = max(0, min(100, health_score - 8))
    growth_health = max(0, min(100, health_score + 2))

    # ==================================================
    # 🧠 AI ONLY FOR TEXT INSIGHT
    # ==================================================

    prompt = f"""
Startup details:

Idea: {data.idea}
Customer: {data.customer}
Pricing: {data.pricing}
Team size: {data.team_size}
Budget: {data.budget}
Situation: {data.situation}

Return ONLY JSON:

{{
 "biggest_problem": "",
 "improvements": ["", "", ""],
 "insight": ""
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    # SAFE JSON EXTRACTION
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1:
        content = content[start:end]

    try:
        ai_result = json.loads(content)
    except:
        ai_result = {
            "biggest_problem": "AI formatting issue",
            "improvements": [
                "Retry request",
                "Check input",
                "Try again"
            ],
            "insight": "Fallback response."
        }

    # FINAL RESULT
    result = {
        "health_score": health_score,
        "market_health": market_health,
        "execution_health": execution_health,
        "finance_health": finance_health,
        "growth_health": growth_health,
        "biggest_problem": ai_result["biggest_problem"],
        "improvements": ai_result["improvements"],
        "insight": ai_result["insight"]
    }

    # SAVE CACHE
    analysis_cache[cache_key] = result
    cache_timestamps[cache_key] = time.time()

    return result