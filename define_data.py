from typing import List, Dict, Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field

 
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_huggingface import HuggingFaceEndpoint
from os

api_key= os.environ.get("api_key")
                        
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=api_key
)

#Define structured outputs
class Plan(BaseModel):
    steps: List[str] = Field(..., description="Short ordered steps for solving the task.")
    key_risks: List[str] = Field(..., description="Major risks/unknowns that should be addressed.")
    desired_output_structure: List[str] = Field(..., description="Headings to include in final answer.")

class Critique(BaseModel):
    issues: List[str] = Field(..., description="Concrete problems with the current draft.")
    missing_points: List[str] = Field(..., description="Important missing considerations.")
    hallucination_risk: List[str] = Field(..., description="Claims that might be risky without sources.")
    score: int = Field(..., ge=0, le=100, description="Overall quality score of the draft.")
    fix_instructions: List[str] = Field(..., description="Actionable steps to improve the draft.")

#Define LangGraph state

class GraphState(TypedDict):
    question: str
    plan: Optional[Dict[str, Any]]
    research_notes: List[str]
    draft: Optional[str]
    critique: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int
