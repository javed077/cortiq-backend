"""
db.py — Supabase client + JWT auth middleware
Place this at: G:\\cortiq\\backend\\db.py
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import jwt as pyjwt
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client

log = logging.getLogger("cortiq")

# ── Supabase client ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set in .env")
    return create_client(url, key)


# ── JWT auth middleware ────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)

def get_user_id(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> Optional[str]:
    if not creds:
        return None
    try:
        payload = pyjwt.decode(
            creds.credentials,
            options={"verify_signature": False},
        )
        user_id = payload.get("sub")
        log.info("auth: user_id=%s", user_id)
        return user_id
    except Exception as e:
        log.warning("auth: JWT decode failed — %s", e)
        return None