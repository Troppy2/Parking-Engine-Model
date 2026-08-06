# TODO — Infrastructure & Production Setup

Things the code already *assumes* exist but that aren't set up yet. Roughly in the
order they'll block you.

---

## 1. Redis (blocking — `weather.py` won't run without it)

`context/weather.py` builds a Redis client from `REDIS_URL` and calls `.get()` / `.setex()`.
There is no Redis server running locally, and Redis has no official native Windows build.

Pick one:

- [ ] **In-memory fallback (easiest, zero install)** — if `REDIS_URL` is unset or the
      connection fails, fall back to a dict-based cache with the same TTL. Keeps the
      Redis code path intact for production; nothing to install to develop.
- [ ] **Docker Desktop** — install, then `docker run -d -p 6379:6379 --name redis redis:7`.
      Matches production exactly. ~1GB install, must be running while you develop.
- [ ] **Hosted free tier (Upstash / Redis Cloud)** — no local install, paste the URL into
      `.env`. Needs an account, and every cache hit is a network round-trip.

## 2. `.env` file (blocking)

Only `.env.example` is committed (`.env` is correctly gitignored). Nothing has an actual
value yet.

- [ ] `cp .env.example .env`
- [ ] `OPENWEATHER_API_KEY` — free tier at openweathermap.org (`/data/2.5/weather` endpoint)
- [ ] `HERE_API_KEY` — free tier at developer.here.com (traffic-flow tiles)
- [ ] `REDIS_URL` — depends on the choice in §1
- [ ] Actually **load** it: `python-dotenv` is in `requirements.txt` but nothing calls
      `load_dotenv()`, so `os.environ["OPENWEATHER_API_KEY"]` will `KeyError` even with a
      populated `.env`.

## 3. Missing dependencies for code that's already designed

- [ ] **Database driver.** `context/events.py` says it "queries `campus_events`", but there's
      no DB driver (`psycopg2` / `sqlalchemy`) in `requirements.txt` and no database anywhere.
      Decide: real Postgres, SQLite for now, or a static seed file?
- [ ] **Web framework.** `engine.py` is described as the "public interface" but there's no
      HTTP layer. If it's meant to be a service, `fastapi` + `uvicorn` need adding.
- [ ] **Pin versions.** Everything is `>=`, so a fresh install can pull a breaking major.
      Pin exact versions or add a lockfile before this is reproducible.

## 4. Containerization / deploy

- [ ] `Dockerfile` for the engine
- [ ] `docker-compose.yml` wiring engine + Redis (+ Postgres if §3 goes that way) so the
      whole stack comes up with one command — this also solves §1 for free
- [ ] Decide where this actually deploys (Railway / Render / Fly / a VM)

## 5. Correctness / robustness gaps

- [ ] **`weather.py:27`** — the `_redis.get(key)` cache read sits *outside* the `try`, so an
      unreachable Redis raises instead of falling back to `_DEFAULT_WEATHER`. Contradicts the
      file's stated "fail soft" intent. Move it inside the `try`.
- [ ] **Imports.** `weather.py` does `from schema import ...` (top-level, not
      `from parking_engine.schema import ...`). This only works if you run from inside
      `parking-engine/`. Add a `pyproject.toml` / `setup.py` and make it a real package —
      note the directory is `parking-engine` (hyphen), which is **not** a valid Python
      module name, so it needs renaming to `parking_engine`.
- [ ] **`__pycache__/schema.cpython-314.pyc` is committed to git.** Add `__pycache__/` and
      `*.pyc` to `.gitignore` and `git rm -r --cached` it.

## 6. Not yet written (stubs — docstring only, no code)

Listed for completeness; these are implementation, not setup.

- [ ] `context/traffic.py` — completely empty
- [ ] `context/events.py`
- [ ] `data/feature_store.py`
- [ ] `engine.py`

## 7. Quality gates (none exist)

- [ ] Any tests at all — `pytest` isn't even in `requirements.txt`
- [ ] CI (GitHub Actions: install, lint, test)
- [ ] Linter/formatter (`ruff` covers both)
