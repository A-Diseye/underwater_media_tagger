---
name: restart-app
description: Restart the local Streamlit app so it picks up changes in tagger/ (Streamlit only hot-reloads the app script). Use after editing anything in tagger/, or when the app shows stale behavior like an unexpected keyword argument error.
---

# Restart the Streamlit app

`uv run streamlit ...` starts two processes: the `uv` wrapper and a Python child. The child holds port 8501, so killing only the wrapper leaves the old code running.

1. Stop both processes:
   ```bash
   pkill -f "streamlit run app/streamlit_app.py"
   ```
2. Confirm nothing is left and the port is free. If a process remains, kill its PID directly:
   ```bash
   pgrep -af "streamlit run" ; ss -ltn | grep 8501
   ```
3. Start it again in the background from the repo root:
   ```bash
   uv run streamlit run app/streamlit_app.py
   ```
4. Wait until the log shows `URL: http://localhost:8501`, then check `curl -s localhost:8501/_stcore/health` returns `ok`.
5. Tell the user to refresh the browser tab.

To only stop the app, do steps 1–2.
