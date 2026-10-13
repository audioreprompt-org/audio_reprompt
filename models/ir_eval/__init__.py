"""IR quality evaluation for the RAG retrieval functions.

Modules (ownership in parentheses):
  data.py         shared loading/sampling utilities           (WS-0)
  diagnostics.py  D1-D5 + anisotropy, exact numpy computation  (WS-A)
  judges.py       relevance judges                             (WS-B)
  metrics.py      IR metrics + bootstrap CIs                   (WS-B)
  retrieval.py    retrieval evaluation loops + baselines       (WS-C)
  report.py       markdown/CSV report                          (WS-C)
  cli.py          command-line entry point                     (WS-C)

Keep this file import-free so unit tests do not need a DB or the encoder.
"""
