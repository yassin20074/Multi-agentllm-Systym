from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, TypedDict
from define_data import graghstate, critique, plan
from Agent_nodes import app, Finalizer

class RequestBody (Basemodel):
    question : str
# ============ FastAPI App ============
app = FastAPI()


@app.post("/run-agent")
def run_agent(body: RequestBody):

    state: GraphState = {
        "question": body.question,
        "plan": None,
        "research_notes": [],
        "draft": None,
        "critique": None,
        "iteration": 0,
        "max_iterations": 2,
    }

    # 1. Plan
    state["plan"] = planner(state["question"])

    # 2. Research
    state["research_notes"] = researcher(state["question"], state["plan"])

    # 3. Draft
    state["draft"] = writer(
        state["question"],
        state["plan"],
        state["research_notes"]
    )

    # 4. Critique
    state["critique"] = critic(state["question"], state["draft"])

    # 5. Final answer
    final = finalizer(
        state["question"],
        state["plan"],
        state["research_notes"],
        state["draft"],
        state["critique"]
    )

    return {
        "plan": state["plan"],
        "research_notes": state["research_notes"],
        "draft": state["draft"],
        "critique": state["critique"],
        "final": final
    }
