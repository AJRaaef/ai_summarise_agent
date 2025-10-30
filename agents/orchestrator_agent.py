# agents/orchestrator_agent.py
"""
OrchestratorAgent: controls pipeline
"""

from agents.semantic_agent import SemanticAgent
from agents.pattern_agent import PatternAgent
from agents.summarizer_agent import SummarizerAgent

class OrchestratorAgent:
    def __init__(self):
        self.semantic = SemanticAgent()
        self.pattern = PatternAgent()
        self.summarizer = SummarizerAgent()

    def run(self, df):
        # 1) semantic inference about columns
        semantic_map = self.semantic.infer_columns(df)

        # 2) pattern recognition & stats
        stats = self.pattern.analyze(df)

        # 3) build NLG summary
        report = self.summarizer.summarize(profile=stats["profile_text"],
                                           semantic_map=semantic_map,
                                           kpis=stats["kpis"],
                                           anomalies=stats["anomalies"],
                                           sample=df.head(10))
        # 4) prepare plots
        plots = stats.get("plots", [])
        return report, plots
