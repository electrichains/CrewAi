import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crewai import Crew, Agent, Task, Process, LLM

app = FastAPI(title="CrewAI API", version="1.15.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunCrewRequest(BaseModel):
    agents: list[dict]
    tasks: list[dict]
    verbose: bool = True


class RunCrewResponse(BaseModel):
    result: str


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.15.5"}


@app.get("/")
async def root():
    return {
        "service": "CrewAI",
        "version": "1.15.5",
        "docs": "/docs",
    }


@app.post("/crew/run", response_model=RunCrewResponse)
async def run_crew(request: RunCrewRequest):
    try:
        agents = [Agent(**a) for a in request.agents]
        tasks = [Task(**t) for t in request.tasks]
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=request.verbose,
        )
        result = crew.kickoff()
        return RunCrewResponse(result=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
