from define_data import plan, llm, critique, stategraph

# Agent nodes (Planner, Researcher, Writer, Critic, finalizer)

PLANNER_SYSTEM = """You are the Planner agent.
Create a concise plan with steps, key risks, and final output headings.
Return valid JSON matching the schema.
"""

RESEARCHER_SYSTEM = """You are the Researcher agent.
You do NOT browse the web. You reason from general knowledge.
Produce bullet research notes covering: cost, speed, privacy, reliability, compliance, vendor lock-in, iteration speed, support.
Keep it practical for startups.
"""

WRITER_SYSTEM = """You are the Writer agent.
Write a structured answer using the plan headings.
Use the research notes.
Be specific, actionable, and include a clear recommendation plus risks.
"""

CRITIC_SYSTEM = """You are the Critic agent.
Review the draft for:
- missing points
- weak reasoning
- overconfidence
- risky claims
Return JSON matching the schema.
"""

FUNALIZER_SYSTEM = """You are the finalizer agent.
Given the plan + research notes + (optional) critique, produce the FINAL answer.
If critique exists, incorporate fixes.
Output must be polished and concise with headings and a confidence score.
"""

def planner_node(state: GraphState) -> GraphState:
    structured_planner = llm.with_structured_output(Plan)
    plan_obj = structured_planner.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=state["question"])
    ])
    state["plan"] = plan_obj.model_dump()
    return state

def researcher_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"Question:\n{state['question']}\n\nPlan:\n{state['plan']}")
    ]).content

    # store as notes (simple split)
    notes = [line.strip("- ").strip() for line in resp.split("\n") if line.strip()]
    state["research_notes"] = notes
    return state

def finalizer(question: str, plan: dict, notes: list, draft: str, critique: dict):
    return llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=f"""
Question: {question}
Plan: {plan}
Notes: {notes}
Draft: {draft}
Critique: {critique}
""")
    ]).content

def writer_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=f"""
        Question:
        {state['question']}

        Plan:
        {state['plan']}

        Research notes:
        {state['research_notes']}

        If critique exists, you may improve the draft accordingly.
        Critique:
        {state.get('critique')}
        """)
            ]).content

    state["draft"] = resp
    return state



def critic_node(state: GraphState) -> GraphState:
    structured_critic = llm.with_structured_output(Critique)
    critique_obj = structured_critic.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"""
    Question:
    {state['question']}

    Draft:
    {state['draft']}
    """)
        ])

    state["critique"] = critique_obj.model_dump()
    state["iteration"] += 1

    return state

from langgraph.graph import StateGraph, END


def should_revise(state: GraphState) -> Literal["revise", "finalize"]:
    score = state["critique"]["score"]

    if state["iteration"] >= state["max_iterations"]:
        return "finalize"

    if score < 80:
        return "revise"

    return "finalize"

# create a graph
workflow = StateGraph(GraphState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("finalizer", finalizer_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")
 
# conditional edge to loop or finalize
workflow.add_conditional_edges(
    "critic",
    should_revise,
    {
        "revise": "writer",
        "finalize": "finalizer",
    }
)

workflow.add_edge("finalizer", END)

app = workflow.compile()



def finalizer_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=FUNALIZER_SYSTEM),
        HumanMessage(content=f"""
Question:
{state['question']}

Plan:
{state['plan']}

Research notes:
{state['research_notes']}

Critique (if any):
{state.get('critique')}

Current draft (if any):
{state.get('draft')}
""")
    ]).content

    state["draft"] = resp
    return state
