import hashlib
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from sqlalchemy.orm import Session

from app.auth.models import User
from app.auth.schemas import Token, UserLogin, UserRegister, UserResponse
from app.auth.utils import create_access_token, hash_password, verify_password
from app.database import get_db

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Helpers — reload .env on every call, read from os.environ directly ─────────
def _reload_env():
    load_dotenv(dotenv_path=_ENV_FILE, override=True)

def _google_client_id()     -> str: _reload_env(); return os.environ.get("GOOGLE_CLIENT_ID", "").strip()
def _google_client_secret() -> str: _reload_env(); return os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
def _google_redirect_uri()  -> str: _reload_env(); return os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback").strip()
def _frontend_url()         -> str: _reload_env(); return os.environ.get("FRONTEND_URL", "http://localhost:8501").strip()
def _secret_key()           -> str: _reload_env(); return os.environ.get("SECRET_KEY", "change-me-in-production").strip()

GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def _google_configured() -> bool:
    cid  = _google_client_id()
    csec = _google_client_secret()
    return bool(
        cid and csec
        and cid  not in ("", "your_google_client_id_here")
        and csec not in ("", "your_google_client_secret_here")
    )


def _make_flow() -> Flow:
    redirect_uri = _google_redirect_uri()
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": _google_client_id(),
                "client_secret": _google_client_secret(),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri],
            }
        },
        scopes=GOOGLE_SCOPES,
        redirect_uri=redirect_uri,
    )


def _cookie_secure(request: Request | None = None) -> bool:
    if request is not None and request.url.scheme == "https":
        return True
    return _google_redirect_uri().startswith("https://")


# ── Standard auth ──────────────────────────────────────────────────────────────
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is inactive")

    token = create_access_token({"sub": user.username, "user_id": user.id})
    return Token(access_token=token, username=user.username, user_id=user.id)


# ── Google OAuth ───────────────────────────────────────────────────────────────
@router.get("/google/login", response_class=HTMLResponse)
def google_login(request: Request):
    """Redirect the browser to Google's consent screen."""
    if not _google_configured():
        return HTMLResponse(_popup_html(
            "error",
            "Google OAuth is not configured.<br><br>"
            "Add <code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> "
            "to your <code>.env</code> file, then restart the server.",
        ))
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # allow http in dev
    flow = _make_flow()
    flow.code_verifier = secrets.token_urlsafe(64)
    auth_url, state = flow.authorization_url(
        prompt="select_account",
        access_type="offline",
        include_granted_scopes="true",
        code_challenge_method="S256",
    )
    response = RedirectResponse(auth_url)
    cookie_secure = _cookie_secure(request)
    response.set_cookie(
        "google_oauth_state",
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=cookie_secure,
    )
    response.set_cookie(
        "google_oauth_code_verifier",
        flow.code_verifier,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=cookie_secure,
    )
    return response


@router.get("/google/callback", response_class=HTMLResponse)
def google_callback(request: Request, db: Session = Depends(get_db)):
    """
    Google redirects here after the user authenticates.
    We exchange the code for user info, create/find the user, issue a JWT,
    then use postMessage to send the token back to the Streamlit popup opener
    and close the popup window.
    """
    if not _google_configured():
        return HTMLResponse(_popup_html("error", "Google OAuth not configured."))

    try:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        saved_state = request.cookies.get("google_oauth_state", "")
        saved_code_verifier = request.cookies.get("google_oauth_code_verifier", "")
        returned_state = request.query_params.get("state", "")

        if not saved_state or not saved_code_verifier:
            return HTMLResponse(_popup_html(
                "error",
                "Google auth failed: missing saved login state. Please try again.",
            ))
        if returned_state != saved_state:
            return HTMLResponse(_popup_html(
                "error",
                "Google auth failed: state mismatch. Please try again.",
            ))

        flow = _make_flow()
        flow.code_verifier = saved_code_verifier
        flow.fetch_token(authorization_response=str(request.url))
        session = flow.authorized_session()
        userinfo = session.get("https://www.googleapis.com/oauth2/v2/userinfo").json()
    except Exception as exc:
        return HTMLResponse(_popup_html("error", f"Google auth failed: {exc}"))

    email    = userinfo.get("email", "")
    name     = userinfo.get("name", "")
    google_id = userinfo.get("id", "")

    if not email:
        return HTMLResponse(_popup_html("error", "Could not retrieve email from Google."))

    # Find or create user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Derive a username from the email local-part, ensure uniqueness
        base_username = email.split("@")[0].replace(".", "_").lower()[:40]
        username = base_username
        counter = 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base_username}{counter}"
            counter += 1

        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(
                hashlib.sha256(f"{google_id}{_secret_key()}".encode("utf-8")).hexdigest()
            ),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    jwt_token = create_access_token({"sub": user.username, "user_id": user.id})
    response = HTMLResponse(_popup_html(
        "success", jwt_token, user.username, user.id, _frontend_url()
    ))
    response.delete_cookie("google_oauth_state")
    response.delete_cookie("google_oauth_code_verifier")
    return response


def _popup_html(status: str, token_or_error: str,
                username: str = "", user_id: int = 0,
                frontend_url: str = "http://localhost:8501") -> str:
    """
    Tiny HTML page served inside the OAuth popup.
    On success: redirects the PARENT window (Streamlit) to a URL containing
                the token as query params, then closes the popup.
    On error:   shows the error with a Close button.
    """
    if status == "success":
        import urllib.parse
        params = urllib.parse.urlencode({
            "g_token": token_or_error,
            "g_user": username,
            "g_uid": user_id,
        })
        redirect_url = f"{frontend_url}?{params}"
        return f"""<!DOCTYPE html>
<html>
<head><title>Signing in…</title>
<style>
  body{{font-family:-apple-system,sans-serif;display:flex;align-items:center;
       justify-content:center;height:100vh;margin:0;background:#111;color:#fff;}}
  .box{{text-align:center;}}
  .spinner{{width:36px;height:36px;border:3px solid #333;border-top-color:#6366f1;
            border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 16px;}}
  @keyframes spin{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>
<div class="box">
  <div class="spinner"></div>
  <p style="color:#aaa;font-size:14px">Signed in — redirecting…</p>
</div>
<script>
  // window.opener may be an iframe (st_components); use .top to reach main Streamlit window
  var target = (window.opener && window.opener.top) ? window.opener.top : window.opener;
  if (target) {{
    try {{
      target.location.href = "{redirect_url}";
    }} catch(e) {{
      // cross-origin restriction fallback: redirect current tab
      window.location.href = "{redirect_url}";
    }}
    setTimeout(() => window.close(), 600);
  }} else {{
    window.location.href = "{redirect_url}";
  }}
</script>
</body>
</html>"""
    else:
        return f"""<!DOCTYPE html>
<html>
<head><title>Auth Error</title>
<style>
  body{{font-family:-apple-system,sans-serif;display:flex;align-items:center;
       justify-content:center;height:100vh;background:#111;color:#fff;margin:0;}}
  .err{{color:#f87171;text-align:center;max-width:400px;padding:24px;}}
  button{{margin-top:16px;padding:8px 24px;background:#6366f1;color:#fff;
          border:none;border-radius:8px;cursor:pointer;font-size:14px;}}
</style>
</head>
<body>
<div class="err">
  <h2>&#x26A0; Authentication Failed</h2>
  <p style="color:#aaa;font-size:13px">{token_or_error}</p>
  <button onclick="window.close()">Close</button>
</div>
</body>
</html>"""
