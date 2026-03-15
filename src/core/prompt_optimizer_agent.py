from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, Field

from src.schemas.evaluation_dataset import EvaluationDataset, GenerationRow


class PromptOptimizationCandidate(BaseModel):
    trace_id: str
    observation_id: str | None = None
    score_name: str | None = None
    score_value: float | None = None
    system_prompt: str | None = None
    generation_text: str | None = None
    recommendation: str


class PromptOptimizationReport(BaseModel):
    average_score: float | None = None
    lowest_scoring_examples: list[PromptOptimizationCandidate] = Field(
        default_factory=list
    )
    highest_scoring_examples: list[PromptOptimizationCandidate] = Field(
        default_factory=list
    )
    summary: str


class PromptOptimizerAgent:
    """Consumes the normalized judge dataset and prepares prompt review candidates."""

    def build_report(
        self, dataset: EvaluationDataset, *, top_k: int = 3
    ) -> PromptOptimizationReport:
        rows = [
            row
            for row in dataset.rows
            if row.judge_scores and (row.system_prompt or row.generation_text)
        ]
        scored_rows = sorted(rows, key=self._row_score)
        average_score = (
            mean(
                [
                    score.score_value
                    for row in rows
                    for score in row.judge_scores
                    if score.score_value is not None
                ]
            )
            if rows
            else None
        )
        lowest = [
            self._candidate_from_row(row, "Review this prompt first; it is underperforming.")
            for row in scored_rows[:top_k]
        ]
        highest = [
            self._candidate_from_row(
                row, "Use this example as a reference when refining prompts."
            )
            for row in scored_rows[-top_k:]
        ]
        return PromptOptimizationReport(
            average_score=average_score,
            lowest_scoring_examples=lowest,
            highest_scoring_examples=list(reversed(highest)),
            summary=self._build_summary(dataset, average_score, len(rows)),
        )

    @staticmethod
    def _row_score(row: GenerationRow) -> float:
        values = [score.score_value for score in row.judge_scores if score.score_value is not None]
        if not values:
            return 0.0
        return mean(values)

    def _candidate_from_row(
        self, row: GenerationRow, recommendation: str
    ) -> PromptOptimizationCandidate:
        best_score = min(
            row.judge_scores,
            key=lambda score: score.score_value
            if score.score_value is not None
            else float("inf"),
        )
        return PromptOptimizationCandidate(
            trace_id=row.trace_id,
            observation_id=row.observation_id,
            score_name=best_score.score_name,
            score_value=best_score.score_value,
            system_prompt=row.system_prompt,
            generation_text=row.generation_text,
            recommendation=recommendation,
        )

    @staticmethod
    def _build_summary(
        dataset: EvaluationDataset, average_score: float | None, candidate_count: int
    ) -> str:
        return (
            f"Dataset contains {dataset.meta.counts.traces} trace(s), "
            f"{dataset.meta.counts.rows} generation row(s), and "
            f"{candidate_count} row(s) with prompt or generation content. "
            f"Average judge score: {average_score if average_score is not None else 'n/a'}."
        )
