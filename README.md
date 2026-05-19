# Multi-Agent LangGraph System (Planner • Researcher • Writer • Critic • Finalizer)

This project implements a multi-agent AI pipeline using LangChain + LangGraph with a structured workflow for solving complex questions.


---

# Overview

**The system is composed of 5 agents:**

1. Planner Agent → Creates a structured plan


2. Researcher Agent → Generates knowledge-based research notes


3. Writer Agent → Produces an initial draft


4. Critic Agent → Evaluates and critiques the draft


5. Finalizer Agent → Produces the final optimized answer




---

# Architecture Flow

User Question
   ↓
Planner → Researcher → Writer → Critic → Finalizer
                                   ↑
                         (Iterative Improvement Loop)


---

# Key Features

Structured outputs using Pydantic models

Multi-agent reasoning pipeline

Iterative critique and improvement loop

HuggingFace LLM integration (Mistral-7B-Instruct)

State management using TypedDict



---

# Technologies Used

LangChain Core

LangGraph (conceptual multi-agent flow)

HuggingFace Hub (Mistral-7B-Instruct-v0.2)

Pydantic (structured outputs)

Python Typing system



---

# Agents Description

**1. Planner Agent**

Input: User question

Output:

Steps

Key risks

Output structure




---

**2. Researcher Agent**

Simulates research using model knowledge

Produces bullet-point insights about:

Cost

Speed

Privacy

Reliability

Compliance

Vendor lock-in




---

**3. Writer Agent**

Builds structured response using:

Plan

Research notes

Previous critique (if any)




---

**4. Critic Agent**

Evaluates draft

Outputs structured feedback:

Issues

Missing points

Risky claims

Score (0–100)

Fix instructions




---

**5. Finalizer Agent**

Produces final polished response

Integrates critique improvements

Adds final clarity and structure



---

State Schema

class GraphState(TypedDict):
    question: str
    plan: Optional[Dict[str, Any]]
    research_notes: List[str]
    draft: Optional[str]
    critique: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int


---
 
# Future Improvements

Add real web search (RAG)

Add memory between runs

Add parallel agent execution

Add tool usage (Python, search, APIs)



---

# Author : eng.yassin sanad.
