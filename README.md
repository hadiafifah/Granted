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
Reviewer mode defaults to `demo` (no export needed).

Optional override (set before starting the server):
```
export GRANT_REVIEW_PROFILE=demo
```
Available values:
- `strict`: conservative reviewer (higher bar to pass)
- `balanced`: default
- `demo`: more lenient scoring while keeping hard fails for expired/unverifiable grants

2. If everything is correct, part of your output should should look like this:
    
    `INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`
    
    `INFO: Started reloader process`
