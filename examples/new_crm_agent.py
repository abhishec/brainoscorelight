"""Example: Build a new CRM agent in <40 lines using BrainOS Core Light.

This replaces ~2,000 lines of boilerplate with 40 lines of business logic.

Run:
    pip install brainos-core-light
    ANTHROPIC_API_KEY=sk-ant-... uvicorn examples.new_crm_agent:app
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from brainos_core import BrainOSWorker

class MyCRMAgent(BrainOSWorker):
    async def handle_domain(self, task_text, task_category, session_id):
        # For non-CRM tasks: generic LLM response
        return await self.executor.call(
            system="You are a helpful business agent. Answer concisely.",
            user=task_text,
            max_tokens=256,
            force_fast=True,
        )

app = FastAPI()
worker = MyCRMAgent(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    mcp_endpoint=os.getenv("GREEN_AGENT_MCP_URL", "http://localhost:9009"),
    fast_model=os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001"),
)

@app.post("/")
async def handle(request: Request):
    body = await request.json()
    params = body.get("params", {})
    message = params.get("message", {})
    task_text = "".join(p.get("text", "") for p in message.get("parts", []))
    task_id = message.get("taskId", "task")
    answer = await worker.handle(task_text)
    return JSONResponse({
        "jsonrpc": "2.0", "id": body.get("id"),
        "result": {
            "kind": "message", "messageId": task_id, "role": "agent",
            "parts": [{"kind": "text", "text": answer}],
        }
    })

@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {"name": "my-crm-agent", "version": "1.0", "skills": ["crm"]}
