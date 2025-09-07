# LLM LangFuse LangChain vibe coded Demo

This repository demonstrates the integration of LangChain with LangFuse to build and monitor large language model (LLM) applications. It showcases how to combine LangChain's capabilities with LangFuse's monitoring tools to create robust and observable LLM applications.

## Overview

This project demonstrates how to integrate LangChain, a framework for building LLM applications, with LangFuse, a tool for monitoring and analyzing LLM interactions. The demo showcases the creation of a simple LLM application that utilizes LangChain's features and is monitored using LangFuse.

## Installation

To set up the project environment, follow these steps:

> This project uses Nvidia GPU with CUDA


**1. Initialize Langfuse:**

```bash
cd langfuse
docker compose up
```

**2. Clone the repository:**

```bash
git clone https://github.com/Bartket/llm-langfuse-langchain-demo.git
cd llm-langfuse-langchain-demo
```
**3. Install dependencies using Poetry:**

```bash
poetry install
```

## Usage

**4. Run the demo application:**

```bash
poetry shell
python demo/langfuse_langchain_demo.py
```