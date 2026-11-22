---
title: General AI Assistant - MCP Client & Server
emoji: 🤖
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
hf_oauth: true
pinned: true
license: apache-2.0
short_description: Solves GAIA Benchmark & Humanity's Last Exam problems
tags:
- building-mcp-track-consumer
- mcp-in-action-track-consumer
---
# General AI Assistant

Prototype **multi-agent AI platform** with high autonomy, including code generation & execution, browser automation, and multi-modal reasoning. The system can solve <a href="https://arxiv.org/pdf/2311.12983">GAIA Benchmark</a> Level 1, 2, and 3 problems as well as <a href="https://arxiv.org/pdf/2501.14249">Humanity's Last Exam</a> problems.

<a href="https://bstraehle.s3.us-west-2.amazonaws.com/demo.mp4">Demo Video</a>

## Architecture

<video controls src="https://bstraehle.s3.us-west-2.amazonaws.com/gaia.mp4"></video>

## Stack

- <a href="https://huggingface.co/">Hugging Face</a> (Spaces & Datasets)
- <a href="https://www.gradio.app/">Gradio</a> (UI, API & MCP Server)
- <a href="https://www.crewai.com/">CrewAI</a> (Agent Framework)
- <a href="https://www.crewai.com/">CrewAI</a> (Observability)
- <a href="https://www.postman.com/ai-engineer/generative-ai-apis/overview">Postman</a> (API & MCP Client - filter by GAIA)</p>

## Models

- <a href="https://ai.google/get-started/our-models/">Gemini</a> (gemini-3.1-pro-preview & gemini-3.8-flash - fallback on daily rate limit)
- <a href="https://platform.openai.com/docs/models">OpenAI</a> (gpt-6.2-sol)
- <a href="https://docs.anthropic.com/en/docs/about-claude/models">Anthropic</a> (claude-sonnet-5-5)

## Leaderboard

[#13 of 10K+](https://huggingface.co/spaces/agents-course/Students_leaderboard)

## Author

[Bernd Straehle](https://huggingface.co/bstraehle)