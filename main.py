from Agent_nodes import app

question="Ai in Robotics"

initial_state: GraphState = {
    "question": question,
    "plan": None,
    "research_notes": [],
    "draft": None,
    "critique": None,
    "iteration": 0,
    "max_iterations": 2,
}



result = app.invoke(initial_state).content
print(result["draft"])
