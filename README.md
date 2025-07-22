# Introduction to AI Engineering (for software engineers)
By Obinna Okechukwu

## What you learn

After completing this book, you will:
- Be able to implement production AI systems with confidence
- Have deep understanding of all major AI APIs
- Have a good understanding of common prompt engineering techniques
- Be able to design and deploy AI agents
- Be able to make informed architectural decisions
- Be able to work proficiently with multimodal AI applications


## Book Structure

### Part 1: Foundations (Chapters 1-8)

#### [Chapter 0: Introduction](chapters/INTRODUCTION.md)

#### [Chapter 1: Welcome to AI engineering](chapters/01-ai-engineering.md)
- What are LLMs and how do they work?
- The paradigm shift: describing vs. instructing
- Your first AI call in Python
- The structure of a conversation (roles: system, user, assistant)
- Practical example: IoT status interpreter
- A simple mental model: tokens, embeddings, prediction
- The landscape of AI models

#### [Chapter 2: Core Concepts - Tokens, Embeddings, and Context](chapters/02-core-concepts.md)
- Tokens and cost calculation
- Using tiktoken to count tokens
- Embeddings and semantic search
- Context windows and memory limitations
- Practical example: semantic search for IoT troubleshooting
- Managing conversation history

#### [Chapter 3: Setting Up Your Development Environment](chapters/03-development-environment.md)
- Python environment setup
- Essential libraries: openai, python-dotenv
- API key security and secret management
- Project structure and virtual environments
- Building a command-line AI chatbot
- Creating requirements.txt

#### [Chapter 4: Understanding AI Capabilities and Limitations](chapters/04-ai-capabilities-limitations.md)
- Core strengths of LLMs
- Common failure modes (hallucinations, math, real-time info)
- Safeguards and best practices
- Building a safe assistant class
- Example: SmartSafeAssistant

### Part 2: API Mastery (Chapters 5-12)

#### [Chapter 5: OpenAI API Complete Guide](chapters/05-openai-api-complete-guide.md)
- Making your first API call
- System messages and personality
- Managing conversation history
- Controlling creativity (temperature, max_tokens)
- Streaming responses
- Function calling
- Vision and audio capabilities
- Example: E-commerce recommendation assistant

#### [Chapter 6: Anthropic Claude API Mastery](chapters/06-anthropic-claude-api-mastery.md)
- Claude API basics and differences from OpenAI
- System prompts and persona control
- Long-context analysis and document Q&A
- Vision and tool use
- Example: IoT fleet management with visual diagnostics
- Model comparison and selection

#### [Chapter 7: Google AI (Gemini) API Guide](chapters/07-google-gemini-api-guide.md)
- Gemini API setup and multimodal capabilities
- Video, audio, and PDF analysis
- Function calling and tool integration
- Building a predictive maintenance system
- Model comparison and integration with Google services

#### [Chapter 8: API Design Patterns and Best Practices](chapters/08-api-design-patterns.md)
- Retry strategies and exponential backoff
- Rate limiting and quotas
- Streaming vs batch responses
- Caching strategies
- Multi-provider failover
- Example: Production-grade IoT command processor

### Part 3: Prompt Engineering Mastery (Chapters 9-14)

#### [Chapter 9: Fundamental Prompt Engineering](chapters/09-fundamental-prompt-engineering.md)
- Anatomy of effective prompts
- Zero-shot, few-shot, and chain-of-thought prompting
- Role-based prompting
- A/B testing prompts
- Practical IoT diagnostic system

#### [Chapter 10: Advanced Prompting Strategies](chapters/10-advanced-prompting-strategies.md)
- Self-consistency for improved reliability
- Tree-of-thought prompting for complex reasoning
- ReAct (Reasoning and Acting) for dynamic problem solving
- Multi-Agent Debate for reducing bias and improving robustness
- Reflexion for iterative self-improvement
- Prompt chaining and workflows

#### [Chapter 11: Structured Output Generation](chapters/11-structured-output-generation.md)
- JSON mode and structured outputs
- Pydantic integration and schema validation
- Constrained code generation
- Template-based code generation
- Example: IoT configuration generator

#### [Chapter 12: Domain-Specific Prompting](chapters/12-domain-specific-prompting.md)
- Technical documentation generation
- Domain-specific code generation and review
- Data analysis and insights
- Customer service and troubleshooting guides

### Part 4: Building AI Agents (Chapters 13-18)

#### [Chapter 13: Introduction to AI Agents](chapters/13-introduction-to-ai-agents.md)
- Chatbots vs. agents
- The Perceive-Think-Act loop
- Agent components: perception, memory, reasoning, action
- Building an autonomous IoT agent

#### [Chapter 14: Tool Use and Function Calling](chapters/14-tool-use-function-calling.md)
- The two-step tool use loop
- Implementing function calling (OpenAI, Claude, Gemini)
- Tool registry and secure execution
- Orchestrating tool chains
- Security and permission management

#### [Chapter 15: Building Production Agents](chapters/15-building-production-agents.md)
- Agent frameworks (LangChain, AutoGen, CrewAI)
- State management and persistence
- Communication protocols
- Observability: logging, metrics, tracing
- Example: Industrial automation agent

#### [Chapter 16: Multi-Agent Systems](chapters/16-multi-agent-systems.md)
- Multi-agent collaboration patterns
- Message passing and coordination
- Consensus mechanisms
- Hierarchical agent structures
- Example: Smart city IoT coordination system

### Part 5: Python Web Applications (Chapters 17-19)

#### [Chapter 17: Building AI-Powered Web Applications with Python](chapters/17-building-ai-powered-web-applications.md)
- Flask and FastAPI for AI endpoints
- File uploads (images, audio, PDFs)
- Streaming responses and SSE
- WebSockets for real-time updates
- Example: IoT device management dashboard

#### [Chapter 18: Real-Time AI Applications](chapters/18-real-time-ai-applications.md)
- WebSocket integration
- Background task processing with Celery
- Caching strategies
- Rate limiting and queue management
- Example: Real-time IoT anomaly detection

#### [Chapter 19: AI Application Architecture Patterns](chapters/19-ai-application-architecture-patterns.md)
- Monolith vs. microservices
- Event-driven architectures
- Database patterns (polyglot persistence)
- Configuration management
- Example: Scalable IoT analytics platform

### Part 6: Production Systems (Chapters 20-25)

#### [Chapter 20: Scaling AI Applications](chapters/20-scaling-ai-applications.md)
- Scaling challenges unique to AI
- Horizontal scaling and load balancing
- Queue-based architectures
- Database design for AI workloads
- Multi-level caching
- Example: Global IoT platform architecture

#### [Chapter 21: Cost Optimization](chapters/21-cost-optimization.md)
- Token usage and cost structure
- Semantic caching
- Model selection strategies
- Batch processing
- Cost monitoring and ROI
- Example: Cost-effective IoT analysis

#### [Chapter 22: Security and Safety](chapters/22-security-safety.md)
- API key management and secret storage
- Prompt injection and input sanitization
- Output filtering and moderation
- Audit logging and compliance
- Example: Secure healthcare IoT system

#### [Chapter 23: Monitoring and Observability](chapters/23-monitoring-observability.md)
- Structured logging
- Metrics and KPIs
- Distributed tracing
- Real-time monitoring dashboards
- Example: IoT system health dashboard

#### [Chapter 24: Testing AI Systems](chapters/24-testing-ai-systems.md)
- Unit and integration testing for AI
- Regression testing with golden datasets
- Load testing AI endpoints
- Example: IoT command validation testing

#### [Chapter 25: Deployment and DevOps](chapters/25-deployment-devops.md)
- CI/CD for AI applications
- Environment management
- Blue-green deployments
- Feature flags for AI features
- Rollback strategies
- Example: IoT firmware update system

### Part 7: Advanced Topics (Chapters 26-30)

#### [Chapter 26: Fine-Tuning and Custom Models](chapters/26-fine-tuning-custom-models.md)
- When to fine-tune vs. prompt engineering or RAG
- Data preparation for fine-tuning
- Running a fine-tuning job
- Evaluating and deploying custom models
- Example: Specialized IoT assistant

#### [Chapter 27: RAG (Retrieval Augmented Generation)](chapters/27-rag-retrieval-augmented-generation.md)
- RAG workflow: indexing, retrieval, generation
- Vector databases and embeddings
- Building a RAG-powered assistant
- RAG vs. fine-tuning
- Example: IoT documentation assistant

#### [Chapter 28: AI Workflows and Orchestration](chapters/28-ai-workflows-orchestration.md)
- From chains to workflows (DAGs)
- Building a workflow orchestrator
- Error handling and retries
- Human-in-the-loop patterns
- Example: Automated IoT incident response

#### [Chapter 29: Emerging Patterns and Future Trends](chapters/29-emerging-patterns-future-trends.md)
- True multimodal AI
- Edge AI deployment
- Federated learning
- Quantum computing and AI
- Example: Next-gen IoT architectures

#### [Chapter 30: Building Your AI Product](chapters/30-building-ai-product.md)
- From technology to solution: product thinking
- The Lean AI Canvas
- MVPs and data flywheels
- Ethical review and safety
- Launching and iterating AI products


## Additional Resources

### General AI Engineering & LLMs
- [GenAI Handbook](https://genai-handbook.github.io/) – A living, textbook-style roadmap for learning modern AI, LLMs, and generative models, with curated links to the best blogs, videos, and courses.
- [The 2025 AI Engineering Reading List (Latent.Space)](https://www.latent.space/p/2025-papers) – A practical, annotated list of 50+ must-read papers, blogs, and models across LLMs, prompting, RAG, agents, codegen, vision, and more.
- [Full End-to-End Pipeline for LLM Apps (Rohan's Bytes)](https://www.rohan-paul.com/p/full-end-to-end-pipeline-for-developing) – A 2024–2025 guide to building, deploying, and monitoring LLM-powered applications, with technical best practices and industry case studies.

### Prompt Engineering
- [Prompt Engineering Guide](https://www.promptingguide.ai/) – Comprehensive guide to prompting techniques, from basics to advanced strategies like chain-of-thought, tree-of-thought, and multi-agent systems.
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) – Official best practices and examples for crafting effective prompts.
- [Prompt Engineering Mastery: The Complete Guide](https://www.aifire.co/p/the-ultimate-guide-for-mastering-ai-engineering-in-2024) – A step-by-step roadmap for mastering prompt engineering and LLMs, with practical resources and project ideas.
- [Anthropic Prompt Engineering Tutorial](https://docs.anthropic.com/claude/docs/prompt-engineering) – Anthropic’s hands-on guide to prompt design for Claude models.

### AI Agents & Autonomous Systems
- [Awesome AI Agents (GitHub)](https://github.com/jim-schwoebel/awesome_ai_agents) – A massive, regularly updated list of 1,500+ resources, tools, frameworks, datasets, and courses for building and learning about AI agents.
- [A Survey on LLM-based Autonomous Agents (arXiv)](https://arxiv.org/abs/2309.07864) – Comprehensive academic survey of agent architectures, tool use, evaluation, and future directions.
- [LangChain Documentation](https://python.langchain.com/docs/) – The most popular open-source framework for building LLM-powered agents, tool use, and RAG systems.
- [LlamaIndex Documentation](https://docs.llamaindex.ai/) – Framework for building data-augmented LLM applications and agentic workflows.

### RAG (Retrieval-Augmented Generation)
- [Deconstructing RAG (LangChain Blog)](https://blog.langchain.dev/deconstructing-rag/) – Practical guide to RAG architectures, vector databases, and best practices.
- [RAG Course (DeepLearning.AI)](https://learn.deeplearning.ai/courses/advanced-rag) – Free video course on advanced RAG techniques and evaluation.

### Benchmarks, Datasets, and Evaluation
- [HELM: Holistic Evaluation of Language Models (Stanford)](https://crfm.stanford.edu/helm/latest/) – A living benchmark for LLMs, covering knowledge, reasoning, safety, and more.
- [Hugging Face Datasets](https://huggingface.co/datasets) – Thousands of open datasets for LLM training, fine-tuning, and evaluation.

### Community & News
- [Latent.Space Newsletter](https://www.latent.space/) – Weekly deep dives and news for AI engineers.
- [AI Fire Academy](https://www.aifire.co/) – Practical guides, workflows, and a community for mastering AI engineering.
---
