from services.hackernews import get_hn_trends
from services.producthunt import get_producthunt_trends
import os

def scan_market():

    hn = get_hn_trends()

    ph = []

    token = os.getenv("PRODUCT_HUNT_TOKEN")

    if token:
        ph = get_producthunt_trends(token)

    trend_score = len(hn) * 2 + len(ph) * 3

    return {
        "trend_score": trend_score,
        "hackernews": hn,
        "producthunt": ph
    }