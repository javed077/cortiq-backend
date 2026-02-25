@app.post("/decision/analyze", response_model=DecisionOutput)
def analyze_startup(data: StartupInput):

    prompt = f"""
Analyze this startup:

Idea: {data.idea}
Customer: {data.customer}
Pricing: {data.pricing}
Team size: {data.team_size}
Budget: {data.budget}
Launch timeline: {data.launch_timeline}

Return JSON:
{{
  "score": integer (0-100),
  "risks": ["risk1","risk2","risk3"],
  "insight": "one short strategic insight"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are Cortiq. Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    content = response.choices[0].message.content
    print(content)

    result = json.loads(content)

    return result