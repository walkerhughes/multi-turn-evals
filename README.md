# Multi-turn Agent Evals

A multi-turn customer onboarding agent built with **LangGraph**, evaluated using the **Harbor framework** on **Modal**. The agent walks users through account creation via a structured conversational flow, and the eval suite validates behavior at both single-step and full-trajectory levels.

## What It Does

The onboarding agent guides users through: greeting, name collection, email collection & verification, plan selection, preference gathering, confirmation, and account creation. Each stage is a LangGraph node with conditional routing between them.

The eval suite tests the agent in two modes:

- **Step-level evals** — inject conversation history and accumulated state, then assert the agent produces the correct next response, stage transition, and tool calls
- **Trajectory-level evals** — drive a full multi-turn conversation and grade on correctness, efficiency, repetition, tone, and edge-case handling

## Project Structure

```
src/
├── agent/              # LangGraph agent
│   ├── graph.py        # StateGraph assembly (nodes, edges, routing)
│   ├── nodes.py        # Node implementations (greeting, collect_name, etc.)
│   ├── state.py        # OnboardingState schema
│   └── tools.py        # Tools (validate_email, send_verification_code, etc.)
├── grading/            # Eval grading logic
│   ├── step_grader.py  # Deterministic step-level grader
│   └── trajectory_grader.py  # Hybrid trajectory grader (deterministic + LLM-as-judge)
├── harbor_agent/       # Harbor BaseAgent wrapper
│   └── onboarding_agent.py
└── scripts/            # Scenario generation & results analysis
    ├── generate_scenarios.py
    └── analyze_results.py
```

## Getting Started

```bash
uv sync          # Install dependencies
make test        # Run all tests
make check       # Lint + typecheck + unit tests
```

## Running Evals

```bash
# Step-level evals (local)
harbor run -c configs/step-eval-job.yaml -n 4

# Trajectory evals (local)
harbor run -c configs/trajectory-eval-job.yaml -n 4

# Full suite on Modal
harbor run -c configs/full-suite-job.yaml --env modal -n 50
```
