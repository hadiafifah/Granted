# Granted

url: https://granted-2o1r.onrender.com/ui

1. From terminal at root directory, run:
```
pip install -r requirements.txt
```
and
```
uvicorn app.main:app --reload --port 8000
```
2. If everything is correct, part of your output should should look like this:
    
    `INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`
    
    `INFO: Started reloader process`

## Google Calendar per-user auth

To add deadlines to each user's own calendar (instead of one shared account), this app now uses OAuth consent in browser:

1. In Google Cloud Console, create an OAuth client and download `credentials.json`.
2. Set authorized redirect URI to:
   - `http://127.0.0.1:8000/auth/google/callback` (local)
   - `https://your-domain/auth/google/callback` (production)
3. Optional env vars:
   - `GOOGLE_OAUTH_CLIENT_PATH` (default: `credentials.json`)
   - `GOOGLE_OAUTH_REDIRECT_URI` (if you want to force a specific callback URL)

In the UI, click **Connect Google Calendar** before asking the agent to create events.
