# FixFirst
FixFirst is a local decision-support app for reviewing unresolved municipal workload.
It uses synthetic educational data and the supplied CSV files; it is not an official city tool.
It ranks locations by current backlog and presents validation results without claiming predictive evidence.

## Key finding
The data behaves like noise: location patterns do not support a reliable forecast, so FixFirst ranks the current backlog and reports the negative validation honestly.

## Backlog score
`backlog_score` combines open incident severity points, 2 points per injury on an open incident, 1 point per unresolved complaint, and 1 extra point per high-priority unresolved complaint. The raw score is scaled from 0 to 100 relative to the highest-scoring location.

## Run
```powershell
pip install -r requirements.txt
python pipeline.py
streamlit run app.py
```

No LLM or API is used.