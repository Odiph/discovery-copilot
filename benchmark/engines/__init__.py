"""Engine adapters, all satisfying the `Engine` protocol in `engines/base.py`.

Nothing outside this package may import a concrete engine module directly except
`run.py`'s dispatch table — score.py and report.py depend only on `EngineResult`.
"""
