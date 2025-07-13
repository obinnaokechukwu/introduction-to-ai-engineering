# Chapter 23: Monitoring and Observability in AI Systems

In traditional software, when something breaks, it often breaks loudly. A server crashes, an endpoint returns a `500` error, a database query times out. You get an alert, you check the logs, and you find the root cause. AI applications, however, can fail silently and insidiously. A model can start "hallucinating" more often, its responses can become subtly less accurate, or a change in user behavior can cause costs to skyrocket without a single error ever being thrown.

This is why **observability** is not just a best practice in AI engineering—it is a fundamental requirement for building and maintaining trustworthy systems. Standard monitoring tells you if your application is *up*; AI observability tells you if your application is *smart*, *safe*, and *sane*. It's the art of understanding the internal state of your AI system from its external outputs.

This chapter will teach you how to instrument your AI applications with the "three pillars of observability"—logging, metrics, and tracing—tailored specifically to the unique challenges of production AI.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Implement structured logging to capture the complete context of every AI interaction.
-   Define and track AI-specific metrics for performance, cost, and quality.
-   Use distributed tracing to follow a request through a complex chain of AI services.
-   Build a real-time monitoring dashboard for an IoT system using Prometheus and Grafana concepts.
-   Set up automated alerts for critical AI-related issues like cost spikes or high hallucination rates.

## The Three Pillars of AI Observability

Modern observability is built on three core types of telemetry data. For AI applications, each has a unique and critical role.

1.  **Logs:** Detailed, timestamped records of discrete events. *AI Twist: A log isn't just an error message; it's the full prompt, the model's response, the tokens used, and the latency of a specific AI call.*
2.  **Metrics:** Aggregated, numerical data measured over time. *AI Twist: Metrics go beyond CPU/memory to include `tokens_per_minute`, `cost_per_user`, `hallucination_rate`, and `cache_hit_ratio`.*
3.  **Traces:** A complete view of a single request as it travels through multiple services. *AI Twist: Tracing is essential for debugging complex prompt chains or multi-agent systems where one user input can trigger multiple, cascading AI calls.*

Let's build a system that incorporates all three.

## Pillar 1: Structured Logging for AI

The first step to understanding your AI is to log every interaction with rich context. The best practice is to use **structured logging**, where log entries are written as JSON objects, not plain text. This makes them machine-readable and easy to query in log management systems like Elasticsearch, Datadog, or Splunk.

We'll use the `structlog` library, which integrates seamlessly with Python's standard `logging` module to produce structured JSON logs.

First, install `structlog`: `pip install structlog`.

### Instrumenting an AI Call with Structured Logging

Let's create a centralized function for making AI calls that automatically logs all the important details.

```python
# structured_logging_example.py
import openai
import os
from dotenv import load_dotenv
import structlog
import uuid
from datetime import datetime

# --- Configure structlog for JSON output ---
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

# --- Load API Key and Initialize Client ---
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def monitored_ai_call(prompt: str, user_id: str, model: str = "gpt-4o-mini"):
    """Makes an AI call with comprehensive structured logging."""
    
    request_id = str(uuid.uuid4())
    start_time = datetime.now()

    log.info(
        "ai_call_start",
        request_id=request_id,
        user_id=user_id,
        model=model,
        prompt_length=len(prompt)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        ai_response_text = response.choices[0].message.content
        usage = response.usage

        log.info(
            "ai_call_success",
            request_id=request_id,
            user_id=user_id,
            model=model,
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            # In production, you might log a hash of the prompt/response for privacy
            prompt=prompt,
            response=ai_response_text
        )
        return ai_response_text
        
    except Exception as e:
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        log.error(
            "ai_call_failure",
            request_id=request_id,
            user_id=user_id,
            model=model,
            latency_ms=round(latency_ms, 2),
            error_message=str(e),
            error_type=type(e).__name__
        )
        raise

# --- Example Usage ---
try:
    monitored_ai_call(
        prompt="Explain the difference between MQTT and HTTP for IoT data.",
        user_id="user-123"
    )
except Exception:
    pass
```

Running this code produces beautifully structured JSON logs that can be easily ingested and analyzed:

```json
{"level": "info", "logger": "__main__", "timestamp": "2024-07-10T14:30:00.123Z", "event": "ai_call_start", "request_id": "...", "user_id": "user-123", "model": "gpt-4o-mini", "prompt_length": 55}
{"level": "info", "logger": "__main__", "timestamp": "2024-07-10T14:30:01.456Z", "event": "ai_call_success", "request_id": "...", "user_id": "user-123", "model": "gpt-4o-mini", "latency_ms": 1333.0, "input_tokens": 15, "output_tokens": 120, "total_tokens": 135, "prompt": "...", "response": "..."}
```

With logs like these, you can easily answer questions like:
-   "Show me all failed API calls for `gpt-4o` in the last hour."
-   "Find all interactions from `user-123` that took longer than 3 seconds."
-   "What was the exact prompt that led to this specific erroneous response?"

## Pillar 2: Metrics for AI Systems

Metrics are numerical data points collected over time. They give you a high-level view of your system's health and performance. For AI applications, we need to track both traditional system metrics and AI-specific ones.

We'll use the `prometheus-client` library, which is the standard for exposing metrics that can be scraped by a **Prometheus** server and visualized in a **Grafana** dashboard.

First, install the library: `pip install prometheus-client`.

### Defining AI-Specific Metrics

Let's define a set of essential metrics for any AI application.

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# --- AI API Metrics ---
AI_API_CALLS_TOTAL = Counter(
    'ai_api_calls_total',
    'Total number of AI API calls.',
    ['model', 'provider', 'status'] # Labels to slice data by
)

AI_API_CALL_DURATION_SECONDS = Histogram(
    'ai_api_call_duration_seconds',
    'Latency of AI API calls.',
    ['model', 'provider']
)

# --- Token and Cost Metrics ---
TOKENS_PROCESSED_TOTAL = Counter(
    'ai_tokens_processed_total',
    'Total tokens processed by the AI.',
    ['model', 'provider', 'direction'] # direction can be 'input' or 'output'
)

ESTIMATED_COST_USD_TOTAL = Counter(
    'ai_estimated_cost_usd_total',
    'Estimated cumulative cost of AI API calls in USD.',
    ['model', 'provider']
)

# --- Quality and Safety Metrics ---
VALIDATION_FAILURES_TOTAL = Counter(
    'ai_validation_failures_total',
    'Total times AI output failed Pydantic or other validation.',
    ['model']
)

CONTENT_SAFETY_FLAGS_TOTAL = Counter(
    'ai_content_safety_flags_total',
    'Total times AI output was flagged by content safety filters.',
    ['model', 'category']
)

# --- Application-Level Metrics ---
ACTIVE_CONVERSATIONS = Gauge(
    'ai_active_conversations',
    'Number of currently active AI conversations.'
)
```

### Instrumenting Our Code with Metrics

Now, let's update our `monitored_ai_call` function to update these metrics with each call.

```python
# monitored_ai_with_metrics.py
# (Includes code from logging example and metrics definitions)

# A simplified cost model for demonstration
MODEL_COSTS = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}} # per 1M tokens

def monitored_ai_call_with_metrics(prompt: str, user_id: str, model: str = "gpt-4o-mini"):
    """Makes an AI call instrumented with structured logging and Prometheus metrics."""
    
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    log.info("ai_call_start", ...) # As before

    # The Histogram context manager automatically times the code block
    with AI_API_CALL_DURATION_SECONDS.labels(model=model, provider='openai').time():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # ... (log success message as before) ...
            
            # --- Update Metrics on Success ---
            AI_API_CALLS_TOTAL.labels(model=model, provider='openai', status='success').inc()
            
            usage = response.usage
            TOKENS_PROCESSED_TOTAL.labels(model=model, provider='openai', direction='input').inc(usage.prompt_tokens)
            TOKENS_PROCESSED_TOTAL.labels(model=model, provider='openai', direction='output').inc(usage.completion_tokens)
            
            cost = ((usage.prompt_tokens / 1_000_000) * MODEL_COSTS[model]['input'] +
                    (usage.completion_tokens / 1_000_000) * MODEL_COSTS[model]['output'])
            ESTIMATED_COST_USD_TOTAL.labels(model=model, provider='openai').inc(cost)
            
            return response.choices[0].message.content
            
        except Exception as e:
            # ... (log error message as before) ...
            
            # --- Update Metrics on Failure ---
            AI_API_CALLS_TOTAL.labels(model=model, provider='openai', status='failure').inc()
            raise
```

To make these metrics accessible, you simply need to run a small HTTP server in your application.

```python
from prometheus_client import start_http_server

# In your application's main entry point:
start_http_server(8000) # Expose metrics on port 8000
```

Now, a Prometheus instance can be configured to scrape the `/metrics` endpoint at `http://localhost:8000`, collecting all this valuable data over time.

## Pillar 3: Distributed Tracing

While metrics give you the "what" (e.g., "latency is high"), **distributed tracing** gives you the "where" and "why." A trace follows a single request through its entire journey across all the microservices and components it touches. This is indispensable for debugging complex AI workflows.

We'll use **OpenTelemetry**, the industry standard for tracing.

First, install the necessary packages: `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`.

### Setting Up a Tracer

You configure a tracer once at the start of your application. It will send trace data to a "collector" (like Jaeger or Honeycomb) which then visualizes it.

```python
# telemetry_setup.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# For demo purposes, we'll print traces to the console.
# In production, you'd use an OTLPSpanExporter to send to a collector.
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)

# Sets the global tracer provider
trace.set_tracer_provider(provider)

# Gets a tracer from the global provider
tracer = trace.get_tracer(__name__)
```

### Instrumenting Our Code with Tracing

We can now use this tracer to create "spans," which represent individual units of work within the trace.

```python
# monitored_ai_with_tracing.py
# (Includes all previous code)

def final_monitored_ai_call(prompt: str, user_id: str, model: str = "gpt-4o-mini"):
    """The final version of our monitored function, now with tracing."""
    
    # tracer.start_as_current_span creates a new span for the trace.
    # The 'with' statement ensures it's properly closed.
    with tracer.start_as_current_span("ai_api_call") as span:
        # Add attributes (metadata) to the span for rich context
        span.set_attribute("ai.model", model)
        span.set_attribute("user.id", user_id)
        span.set_attribute("prompt.length", len(prompt))
        
        # Now, call our existing function that handles logging and metrics
        try:
            result = monitored_ai_call_with_metrics(prompt, user_id, model)
            
            # Add results to the span
            span.set_attribute("response.length", len(result))
            span.set_status(trace.Status(trace.StatusCode.OK))
            
            return result
        except Exception as e:
            # Record the exception in the trace for easy debugging
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
```

When you run this code, OpenTelemetry will print a JSON representation of the trace to your console, showing the duration, attributes, and status of the AI call. In a production system with multiple services, traces from each service would be linked together by a common `trace_id`, giving you a complete, end-to-end view of the request.

## Practical Project: A Real-Time IoT Monitoring Dashboard

Let's build a simple but complete application that brings all three pillars together. We'll create a FastAPI service that simulates receiving IoT data, analyzes it with our fully-monitored AI function, and exposes a metrics endpoint for Prometheus.

```python
# complete_monitoring_app.py
from fastapi import FastAPI, BackgroundTasks
from prometheus_client import make_asgi_app
import random
import asyncio

# Import all our previously defined functions and classes:
# - setup_logging, setup_metrics, setup_tracing
# - monitored_ai_call_with_logging_metrics_and_tracing (our final instrumented function)

# --- FastAPI Application ---
app = FastAPI()

# Add Prometheus metrics endpoint to the app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/iot/data")
async def process_iot_data(data: dict, background_tasks: BackgroundTasks):
    """Receives IoT data and triggers an AI analysis in the background."""
    device_id = data.get("device_id", "unknown")
    
    prompt = f"Analyze this IoT sensor data and determine if it indicates an anomaly. Data: {data}"
    
    # Run the expensive AI call in the background to keep the API responsive
    background_tasks.add_task(final_monitored_ai_call, prompt, user_id=device_id)
    
    return {"status": "accepted", "message": "Data received and queued for analysis."}

async def simulate_iot_traffic():
    """A background task to continuously generate sample IoT data."""
    while True:
        device_id = f"sensor-{random.randint(1, 10)}"
        data = {
            "device_id": device_id,
            "temperature": round(70 + random.uniform(-5, 15), 2),
            "humidity": round(50 + random.uniform(-10, 10), 2)
        }
        
        # Simulate making a request to our own API
        # (In a real system, this would come from an external source)
        try:
            # We don't need the result here, just to trigger the background task
            await process_iot_data(data, BackgroundTasks())
        except Exception:
            pass
            
        await asyncio.sleep(random.uniform(0.5, 2.0))

@app.on_event("startup")
async def startup_event():
    # Start the simulation traffic in the background
    asyncio.create_task(simulate_iot_traffic())

# To run: uvicorn complete_monitoring_app:app --reload
# Then visit http://localhost:8000/metrics to see your Prometheus metrics.
# Your console will show the structured JSON logs and traces.
```

With this application running, you have a complete observability solution:
-   **Logs:** Every AI call produces a structured JSON log on the console.
-   **Metrics:** The `/metrics` endpoint provides real-time data ready for Prometheus.
-   **Traces:** The console will show detailed traces for each AI interaction.

You now have the visibility needed to confidently run, debug, and scale your AI applications in production.

# References and Further Reading

- [What You Actually Need to Monitor AI Systems in Production (Sentry, 2025)](https://blog.sentry.io/what-you-actually-need-to-monitor-ai-systems-in-production/)
- [Chapter 12 - Observability (Azure AI in Production Guide)](https://azure.github.io/AI-in-Production-Guide/chapters/chapter_12_keeping_log_observability)
- [Production-Ready Observability Platform for AI Systems (Medium, 2023)](https://medium.com/@bijit211987/production-ready-observability-platform-for-ai-systems-17923d19639b)
- [A Simple Recipe for LLM Observability (Medium, 2025)](https://statistician-in-stilettos.medium.com/a-simple-recipe-for-llm-observability-5c6f24c96310)
- [AI Observability: A Complete Guide to Monitoring Model Performance in Production (Galileo AI, 2025)](https://www.galileo.ai/blog/ai-observability)