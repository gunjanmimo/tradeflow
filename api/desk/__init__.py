"""
The trade desk (desk/desk.py): before any position opens, an observer watches the
stock and two LLM agents -- an analyst and a critic, run locally through Ollama --
argue the trade. The executor refuses a buy the desk has not approved.
"""
