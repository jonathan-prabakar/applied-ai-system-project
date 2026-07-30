"""
Agentic recommendation workflow: PLAN -> ACT -> CHECK -> FIX (repeat).

This is the "agentic" upgrade to the music recommender. Instead of scoring the
catalog once and printing whatever comes out, the system runs a multi-step
reasoning chain in which it CALLS NAMED TOOLS (see src/tools.py), reads back the
results, and decides what to do next:

    1. PLAN  - call catalog_stats / validate_energy / detect_preference_conflict
               to understand and sanitise the request, then choose a strategy.
    2. ACT   - call the recommender (recommend_songs) with the current strategy.
    3. CHECK - call count_genres and inspect scores to grade its own output
               against quality guardrails. This is the agent "testing its work".
    4. FIX   - if a check fails, adjust the strategy and loop back to ACT.

Every thought, tool-call, observation, and decision is recorded as a
ReasoningStep, so the full chain can be printed and saved to a committed log
(see AgentResult.to_markdown / write_trace and ai_interactions.md).

This mirrors a coding assistant that writes code, runs the tests, reads the
failures, and edits until the tests pass. The loop is bounded (guardrail against
infinite fixing), every step is logged, and bad input is handled safely.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src import tools
from src.recommender import DEFAULT_WEIGHTS, recommend_songs

logger = logging.getLogger("music_agent")

# ---- Guardrails / tunables ------------------------------------------------
MAX_ITERATIONS = 4          # hard cap: never loop forever while "fixing"
MIN_TOP_SCORE = 3.0         # a good top pick should clear this bar
MAX_SAME_GENRE = 2          # at most this many of one genre in the shortlist

Recommendation = Tuple[Dict, float, str]


@dataclass
class ReasoningStep:
    """One link in the agent's decision chain: think -> (tool call) -> decide."""
    stage: str                        # PLAN / ACT / CHECK / FIX
    thought: str                      # why the agent is doing this
    tool: Optional[str] = None        # name of the tool called, if any
    tool_input: Any = None            # what was passed to the tool
    observation: Any = None           # what the tool returned
    decision: str = ""                # what the agent concluded / did next


@dataclass
class AgentResult:
    """The full trace of an agent run, so callers can inspect *how* it decided."""
    recommendations: List[Recommendation]
    iterations: int
    strategy: Dict
    trace: List[str] = field(default_factory=list)          # short human lines
    reasoning: List[ReasoningStep] = field(default_factory=list)  # full chain
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self, header: str = "") -> str:
        """Render the reasoning chain as a Markdown block for ai_interactions.md."""
        lines: List[str] = []
        if header:
            lines.append(f"### {header}")
            lines.append("")
        for i, step in enumerate(self.reasoning, start=1):
            lines.append(f"**Step {i} — {step.stage}**")
            lines.append(f"- 🧠 Thought: {step.thought}")
            if step.tool:
                lines.append(f"- 🔧 Tool call: `{step.tool}({_short(step.tool_input)})`")
                lines.append(f"- 👁️ Observation: `{_short(step.observation)}`")
            if step.decision:
                lines.append(f"- ✅ Decision: {step.decision}")
            lines.append("")
        lines.append(f"_Settled after {self.iterations} iteration(s)._")
        if self.warnings:
            lines.append("")
            lines.append("_Guardrail notes:_ " + "; ".join(self.warnings))
        return "\n".join(lines)


class AgenticRecommender:
    """Runs the plan/act/check/fix tool-calling loop over the recommender."""

    def __init__(self, songs: List[Dict], k: int = 5):
        if not songs:
            raise ValueError("AgenticRecommender needs a non-empty song catalog.")
        self.songs = songs
        self.k = k

    # -- public entry point --------------------------------------------------
    def run(self, user_prefs: Dict) -> AgentResult:
        result = AgentResult(recommendations=[], iterations=0, strategy={})

        # ---------- PLAN ----------
        prefs, strategy = self._plan(user_prefs, result)
        result.strategy = strategy

        # ---------- ACT / CHECK / FIX loop ----------
        for iteration in range(1, MAX_ITERATIONS + 1):
            result.iterations = iteration

            # ACT
            recs = recommend_songs(
                prefs,
                self.songs,
                k=self.k,
                weights=strategy["weights"],
                diversity_penalty=strategy["diversity_penalty"],
            )
            result.recommendations = recs
            top = recs[0][1] if recs else 0.0
            logger.info("ACT iter=%d produced %d recs, top score=%.2f",
                        iteration, len(recs), top)
            result.trace.append(f"ACT #{iteration}: {len(recs)} recs, top score {top:.2f}")
            result.reasoning.append(ReasoningStep(
                stage="ACT",
                thought=f"Score the catalog with the current strategy "
                        f"({_fmt_strategy(strategy)}).",
                tool="recommend_songs",
                tool_input={"k": self.k,
                            "diversity_penalty": strategy["diversity_penalty"]},
                observation={"n": len(recs), "top_score": round(top, 2)},
                decision=f"Produced {len(recs)} candidate recommendations.",
            ))

            # CHECK
            issues = self._check(recs, result)
            if not issues:
                logger.info("CHECK iter=%d passed", iteration)
                result.trace.append(f"CHECK #{iteration}: PASSED")
                break
            logger.warning("CHECK iter=%d found issues: %s", iteration, issues)
            result.trace.append(f"CHECK #{iteration}: {len(issues)} issue(s) -> "
                                + "; ".join(i["message"] for i in issues))

            # FIX (unless this was the last allowed iteration)
            if iteration == MAX_ITERATIONS:
                msg = ("stopped after MAX_ITERATIONS with unresolved issues; "
                       "returning best effort")
                logger.warning(msg)
                result.trace.append(f"FIX: {msg}")
                result.warnings.append(msg)
                result.reasoning.append(ReasoningStep(
                    stage="FIX", thought="Hit the iteration guardrail.",
                    decision=msg))
                break

            changed = self._fix(strategy, issues, result)
            if not changed:
                logger.info("FIX iter=%d: no further adjustment possible", iteration)
                result.trace.append("FIX: no further adjustment possible; stopping")
                result.reasoning.append(ReasoningStep(
                    stage="FIX", thought="Reviewed the issues.",
                    decision="No further strategy adjustment possible; stopping."))
                break
            result.trace.append(f"FIX #{iteration}: {changed}")

        return result

    # -- PLAN ----------------------------------------------------------------
    def _plan(self, user_prefs: Dict, result: AgentResult) -> Tuple[Dict, Dict]:
        """Sanitise input via tool-calls and pick a starting strategy.

        Never mutates the caller's dict.
        """
        prefs = dict(user_prefs) if user_prefs else {}

        # Tool call 1: understand the catalog before recommending.
        stats = tools.catalog_stats(self.songs)
        result.reasoning.append(ReasoningStep(
            stage="PLAN",
            thought="Before recommending, understand what's in the catalog.",
            tool="catalog_stats", tool_input={"n": len(self.songs)},
            observation={"n_songs": stats["n_songs"], "n_genres": stats["n_genres"]},
            decision=f"Catalog has {stats['n_songs']} songs across "
                     f"{stats['n_genres']} genres.",
        ))

        if not prefs:
            warn = "empty preferences; falling back to neutral defaults"
            result.warnings.append(warn)
            result.reasoning.append(ReasoningStep(
                stage="PLAN", thought="The request has no preferences.",
                decision=warn))

        # Tool call 2: validate/clamp the energy value.
        if "energy" in prefs:
            check = tools.validate_energy(prefs["energy"])
            result.reasoning.append(ReasoningStep(
                stage="PLAN",
                thought="Energy must be on a 0-1 scale; verify the request.",
                tool="validate_energy", tool_input=prefs["energy"],
                observation=check,
                decision=(check["message"] if check["changed"]
                          else "energy is valid; no change"),
            ))
            prefs["energy"] = check["value"]
            if check["changed"]:
                result.warnings.append(check["message"])

        strategy = {
            "weights": copy.deepcopy(DEFAULT_WEIGHTS),
            "diversity_penalty": 0.0,
        }

        # Tool call 3: detect an internally conflicting request.
        conflict = tools.detect_preference_conflict(prefs)
        if conflict["conflict"]:
            strategy["weights"]["mood"] *= 0.75
            warn = f"conflicting request ({conflict['reason']}); softening mood weight"
            result.warnings.append(warn)
            result.reasoning.append(ReasoningStep(
                stage="PLAN",
                thought="Check whether mood and energy contradict each other.",
                tool="detect_preference_conflict", tool_input=dict(prefs),
                observation=conflict,
                decision="Softened mood weight by 25% so the agent doesn't "
                         "over-commit to mood before CHECK runs.",
            ))
        else:
            result.reasoning.append(ReasoningStep(
                stage="PLAN",
                thought="Check whether mood and energy contradict each other.",
                tool="detect_preference_conflict", tool_input=dict(prefs),
                observation=conflict,
                decision="No conflict; keep default weights.",
            ))

        logger.info("PLAN complete: strategy=%s warnings=%s",
                    strategy, result.warnings)
        result.trace.append(f"PLAN: strategy={_fmt_strategy(strategy)}")
        for w in result.warnings:
            result.trace.append(f"PLAN: guardrail note - {w}")
        return prefs, strategy

    # -- CHECK ---------------------------------------------------------------
    def _check(self, recs: List[Recommendation], result: AgentResult) -> List[Dict]:
        """Grade the current recommendations using tool-calls. Returns issues."""
        issues: List[Dict] = []

        if not recs:
            issues.append({"type": "empty", "message": "no recommendations produced"})
            result.reasoning.append(ReasoningStep(
                stage="CHECK", thought="Inspect the result set.",
                decision="No recommendations were produced — flagging."))
            return issues

        # Relevance: is the top pick actually a decent match?
        top_score = recs[0][1]
        if top_score < MIN_TOP_SCORE:
            issues.append({
                "type": "low_relevance",
                "message": f"top score {top_score:.2f} below bar {MIN_TOP_SCORE}",
            })

        # Diversity: call the count_genres tool and look for domination.
        counts = tools.count_genres(recs)
        dominant = max(counts, key=counts.get)
        if counts[dominant] > MAX_SAME_GENRE:
            issues.append({
                "type": "low_diversity",
                "message": (f"genre '{dominant}' appears {counts[dominant]}x "
                            f"(> {MAX_SAME_GENRE})"),
            })

        result.reasoning.append(ReasoningStep(
            stage="CHECK",
            thought="Grade the shortlist: is the top pick relevant, and is any "
                    "one genre over-represented?",
            tool="count_genres", tool_input={"n_recs": len(recs)},
            observation={"top_score": round(top_score, 2), "genre_counts": counts},
            decision=("All checks passed." if not issues
                      else "Issues: " + "; ".join(i["message"] for i in issues)),
        ))
        return issues

    # -- FIX -----------------------------------------------------------------
    def _fix(self, strategy: Dict, issues: List[Dict],
             result: AgentResult) -> Optional[str]:
        """Adjust the strategy in place to address issues. Returns a description
        of what changed, or None if nothing could be adjusted."""
        actions: List[str] = []
        for issue in issues:
            if issue["type"] == "low_diversity":
                new_penalty = strategy["diversity_penalty"] + 1.5
                strategy["diversity_penalty"] = new_penalty
                actions.append(f"raised diversity_penalty to {new_penalty:.1f}")
            elif issue["type"] == "low_relevance":
                strategy["weights"]["genre"] += 0.5
                strategy["weights"]["mood"] += 0.5
                actions.append("boosted genre/mood weights by 0.5 to sharpen relevance")

        summary = "; ".join(actions) if actions else None
        if summary:
            result.reasoning.append(ReasoningStep(
                stage="FIX",
                thought="A check failed; adjust the strategy and retry.",
                decision=summary,
            ))
        return summary


def write_trace(results: List[Tuple[str, AgentResult]], path: str) -> None:
    """Write one or more labelled agent runs to a committed Markdown log file."""
    blocks = ["# Agent Reasoning Trace Log", "",
              "_Auto-generated by `src/agent.py`. Each section is the full "
              "tool-calling decision chain for one request._", ""]
    for label, res in results:
        blocks.append(res.to_markdown(header=label))
        blocks.append("\n---\n")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).rstrip() + "\n")


# ---- small formatting helpers ---------------------------------------------
def _fmt_strategy(strategy: Dict) -> str:
    w = strategy["weights"]
    return (f"weights(g={w['genre']:.1f},m={w['mood']:.1f},e={w['energy']:.1f},"
            f"d={w['danceability']:.1f},a={w['acoustic']:.1f}) "
            f"diversity={strategy['diversity_penalty']:.1f}")


def _short(value: Any, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
