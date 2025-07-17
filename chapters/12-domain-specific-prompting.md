# Chapter 12: Domain-Specific Prompting

So far, we have explored universal prompting strategies that work across a wide range of tasks. However, to unlock the highest levels of performance and create truly expert-level AI systems, we must move from general instructions to **domain-specific prompting**. This involves embedding deep, industry-specific knowledge, terminology, and workflows directly into your prompts.

Think of it as the difference between hiring a generalist and hiring a specialist. A generalist can give you a decent overview, but a specialist—a manufacturing engineer, a biomedical data analyst, a cybersecurity expert—will provide insights that are more nuanced, relevant, and actionable because they understand the specific context, constraints, and objectives of their field. This chapter will teach you how to "inject" that expertise into your AI.

### Learning Objectives

By the end of this chapter, you will be able to:

- Craft prompts tailored to specific industries like manufacturing, healthcare, and finance.
- Generate high-quality, domain-specific technical documentation.
- Build sophisticated code generation workflows that adhere to industry best practices.
- Design AI-powered data analysis systems that understand domain-specific KPIs.
- Develop effective, context-aware customer service automation for specialized products.
- Create detailed IoT device troubleshooting guides using embedded domain knowledge.

## The Power of Context: Generic vs. Domain-Specific Prompts

The core idea behind domain-specific prompting is to provide the AI with the rich context it needs to reason like an expert in a particular field. Let's see a direct comparison.

### A Generic Analysis

A generic prompt asks a general question and receives a general answer.

```python
import openai

client = openai.OpenAI()

def generic_analysis(data: str) -> str:
    """A generic prompt that lacks specific domain context."""
    prompt = f"Analyze this data and provide insights: {data}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### A Domain-Specific Analysis

A domain-specific prompt imbues the AI with a persona and asks it to consider factors relevant to that domain.

```python
def domain_specific_analysis(data: str, domain: str) -> str:
    """A prompt that provides the AI with a specific expert role and context."""
    
    # Define expert personas for different domains
    domain_personas = {
        "manufacturing": "You are a manufacturing process engineer with 15 years of experience in lean manufacturing and quality control.",
        "healthcare": "You are a biomedical data analyst specializing in patient monitoring device data and regulatory compliance."
    }
    persona = domain_personas.get(domain, "You are a technical expert.")
    
    prompt = f"""
As {persona}, analyze the following operational data.

Data:
{data}

Provide your analysis, focusing specifically on:
- Industry-specific Key Performance Indicators (KPIs).
- Potential safety and regulatory implications.
- Opportunities for operational efficiency improvements relevant to the {domain} sector.
- Actionable insights tailored to a {domain} context.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
```

Let's compare the outputs using some sample factory data.

```python
factory_data = """
Production Line 3 Status:
- Output: 480 units/hour (target: 500)
- Defect Rate: 1.5% (target: <1%)
- Motor Temperature: 85°C (safe limit: 90°C)
"""

print("--- Generic Analysis ---")
print(generic_analysis(factory_data))

print("\n--- Manufacturing-Specific Analysis ---")
print(domain_specific_analysis(factory_data, "manufacturing"))
```

**Analysis Results:**

The generic analysis will correctly note that output is below target and the defect rate is high. The manufacturing-specific analysis, however, will likely discuss concepts like Overall Equipment Effectiveness (OEE), suggest potential causes for the reduced throughput related to motor temperature, and frame the defect rate in terms of Six Sigma or other quality control methodologies. It speaks the language of the domain.

## Domain 1: Generating Technical Documentation

AI is a powerful tool for generating documentation, but to create truly useful docs, you must provide it with the right technical and audience context.

### Generating API Documentation

Let's generate developer-focused documentation for an IoT API.

```python
import json
import openai

client = openai.OpenAI()

def generate_api_documentation(api_spec: dict) -> str:
    """Generates comprehensive API documentation from a specification."""
    
    prompt = f"""
You are a senior technical writer specializing in creating clear, developer-friendly API documentation.

Generate a complete API reference document in Markdown format for the following API specification.

**API Specification:**

{json.dumps(api_spec, indent=2)}


**Documentation Requirements:**
1. **Overview:** Explain the API's purpose, base URL, version, and authentication method.
2. **Endpoints:** For each endpoint, detail the HTTP method, path, description, parameters, request body (if any), and example responses for both success (200 OK) and common errors (e.g., 404 Not Found, 401 Unauthorized).
3. **Code Examples:** Provide practical code examples for each endpoint in Python (using the `requests` library) and a `cURL` command.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# Define a simple specification for our IoT API
iot_api_specification = {
    "name": "IoT Device Management API",
    "version": "v1.2",
    "baseURL": "https://api.myiot.com/v1",
    "authentication": "API Key in 'X-API-Key' header",
    "endpoints": [
        {
            "method": "GET",
            "path": "/devices/{device_id}",
            "description": "Retrieves the status and metadata for a specific device."
        },
        {
            "method": "POST",
            "path": "/devices/{device_id}/commands",
            "description": "Sends a command (e.g., 'reboot', 'update_firmware') to a device."
        }
    ]
}

api_docs = generate_api_documentation(iot_api_specification)
print("--- Generated API Documentation ---")
print(api_docs)
```

By providing the full specification and clear instructions on the required sections, we get a complete and professional document, not just a simple description.

## Domain 2: Code Generation and Review

As we've seen, AI can write code. But to write *good* code, it needs to understand the patterns and constraints of the specific domain (e.g., web development, data processing, embedded systems).

### Generating Code with Domain-Specific Patterns

Let's generate a Python class for an MQTT client, a common component in IoT systems, and enforce IoT-specific best practices.

```python
def generate_iot_code(requirements: dict) -> str:
    """Generates code that follows specific IoT domain patterns."""
    
    # Define best practices for this domain
    iot_patterns = [
        "Implement automatic reconnection logic with exponential backoff.",
        "Handle potential network errors and connection losses gracefully.",
        "Use comprehensive logging for easy debugging in the field.",
        "Ensure all network operations are asynchronous (`async`/`await`).",
        "Include methods for clean connection teardown."
    ]
    
    prompt = f"""
You are an expert embedded systems developer specializing in IoT communication protocols.
Generate a production-ready Python class for an MQTT client based on these requirements.

**Requirements:**
{json.dumps(requirements, indent=2)}

**Mandatory Implementation Patterns:**
- {chr(10).join(f"- {p}" for p in iot_patterns)}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o", # Use a more powerful model for better code
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

mqtt_client_requirements = {
    "purpose": "Collects sensor data from topics and forwards it to a cloud API.",
    "broker_address": "mqtt.industrial-iot.com",
    "topics_to_subscribe": ["factory/+/temperature", "factory/+/pressure"],
    "authentication": "Username/Password"
}

generated_code = generate_iot_code(mqtt_client_requirements)
print("--- Generated IoT MQTT Client Code ---")
print(generated_code)
```

By explicitly listing the required engineering patterns, you guide the AI to produce code that is not just functional but also robust and maintainable in a real-world IoT environment.

### Performing a Domain-Specific Code Review

We can also use AI to review code, asking it to look for problems specific to a domain.

```python
def review_iot_code(code_snippet: str) -> str:
    """Performs a code review with a focus on IoT-specific issues."""
    
    iot_review_criteria = [
        "Correct handling of device disconnections and reconnections.",
        "Efficient use of network bandwidth and power, especially for battery-powered devices.",
        "Security of the communication channel (e.g., use of TLS).",
        "Robust error handling for intermittent connectivity.",
        "Scalability to handle hundreds or thousands of devices."
    ]
    
    prompt = f"""
You are a senior IoT architect performing a code review.
Analyze the following Python code for an IoT application.

**Code to Review:**
{code_snippet}


**Review Focus:**
Evaluate the code against these IoT-specific best practices:
- {chr(10).join(f"- {c}" for c in iot_review_criteria)}

Provide your review in a structured format with sections for 'Strengths', 'Critical Issues', and 'Suggestions for Improvement'.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# A simple (and flawed) code snippet to be reviewed
sample_code = """
import paho.mqtt.client as mqtt
import time

def on_message(client, userdata, msg):
    print(f"Received data: {msg.payload.decode()}")

client = mqtt.Client()
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60)
client.subscribe("sensors/temperature")

while True:
    client.loop()
    time.sleep(1)
"""

code_review = review_iot_code(sample_code)
print("\n--- Domain-Specific Code Review ---")
print(code_review)
```

## Domain 3: Data Analysis and Insights

When analyzing data, providing domain context transforms the AI from a simple data processor into a insightful analyst.

### Industrial Manufacturing Analytics

Let's analyze production data from the perspective of a quality control engineer.


```python
def analyze_manufacturing_data(data: dict) -> str:
    """Analyzes production data with a manufacturing expert persona."""
    
    prompt = f"""
You are a Quality Control Engineer in a manufacturing plant. Analyze the following production report.

**Production Data:**
{json.dumps(data, indent=2)}

Provide your analysis focusing on:
- **Overall Equipment Effectiveness (OEE):** Calculate or estimate the OEE score.
- **Root Cause Analysis:** What is the most likely cause of the increased defect rate?
- **Actionable Insights:** What specific, practical steps should be taken to improve quality and throughput?
- **Process Control:** Are the processes in a state of statistical control?
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

production_data = {
    "shift_duration_hours": 8,
    "planned_downtime_minutes": 30,
    "unplanned_downtime_minutes": 45,
    "ideal_cycle_time_seconds": 60,
    "total_units_produced": 400,
    "defective_units": 20
}

manufacturing_analysis = analyze_manufacturing_data(production_data)
print("--- Manufacturing Data Analysis ---")
print(manufacturing_analysis)
```

By framing the request this way, the AI will use domain-specific concepts like OEE and provide recommendations that are relevant to a factory floor manager, rather than just stating that the numbers are low.

## A Practical Project: Domain-Specific IoT Troubleshooting

Let's build a troubleshooting assistant that leverages deep domain knowledge about different types of IoT devices to provide expert-level guidance.

```python
class IoTTroubleshootingSystem:
    def __init__(self):
        self.client = openai.OpenAI()
        # This knowledge base could be loaded from a database or configuration file
        self.device_knowledge = {
            "pressure_transmitter": {
                "common_issues": ["Clogged impulse lines", "Diaphragm failure", "Calibration drift due to temperature changes"],
                "diagnostic_tools": ["Pressure calibrator", "Manometer", "Multimeter"],
                "safety_note": "Always isolate the device from process pressure before disconnecting."
            },
            "vibration_sensor": {
                "common_issues": ["Improper mounting", "Electromagnetic interference from motors", "Sensor overload"],
                "diagnostic_tools": ["Vibration analyzer", "Oscilloscope", "Torque wrench"],
                "safety_note": "Be aware of rotating machinery when approaching the sensor."
            }
        }

    def generate_troubleshooting_guide(self, device_type: str, issue_description: str) -> str:
        """Generates a detailed, domain-specific troubleshooting guide."""
        
        # Retrieve domain knowledge for the specific device type
        knowledge = self.device_knowledge.get(device_type, {})
        
        prompt = f"""
You are an expert field service technician creating a troubleshooting guide.

**Device Type:** {device_type}
**Reported Issue:** {issue_description}

**Your Knowledge Base for this Device Type:**
- Common Issues: {knowledge.get('common_issues', 'N/A')}
- Required Tools: {knowledge.get('diagnostic_tools', 'N/A')}
- Critical Safety Note: {knowledge.get('safety_note', 'Follow standard safety procedures.')}

Create a systematic, step-by-step troubleshooting guide. Structure it with the following sections:
1. **Initial Assessment & Safety:** What to check first and what safety precautions to take.
2. **Hypothesis 1 (Most Likely Cause):** State the most probable cause based on the issue and your knowledge.
3. **Verification Steps for Hypothesis 1:** List the specific actions to confirm or deny this cause.
4. **Hypothesis 2 (Next Likely Cause):** State the second most probable cause.
5. **Verification Steps for Hypothesis 2:** List the actions to test this hypothesis.
6. **Escalation Path:** When to stop and call for a specialist engineer.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

# --- Demo of the troubleshooting system ---
troubleshooter = IoTTroubleshootingSystem()
reported_problem = "The pressure transmitter is showing erratic readings that jump up and down, but the system pressure seems stable."

guide = troubleshooter.generate_troubleshooting_guide("pressure_transmitter", reported_problem)
print("--- Generated Troubleshooting Guide ---")
print(guide)
```

This final example shows how embedding a structured knowledge base directly into the prompt allows the AI to generate highly relevant, expert-level content that is far more useful than a generic response.

## Conclusion

Domain-specific prompting elevates you from a user of AI to an architect of AI-powered expertise. By embedding the context, vocabulary, and workflows of a specific field into your prompts, you can create applications that deliver specialized and significant value.

Remember these key best practices:
- **Give the AI a Persona:** Start your prompt by telling the AI who it is ("You are a...").
- **Provide Domain Context:** Include industry-specific rules, constraints, and knowledge.
- **Use Domain Terminology:** Speak the language of the expert you want the AI to emulate.
- **Structure for the Domain:** Ask for outputs that match the formats used in that field (e.g., API docs, engineering reports, financial analyses).

By applying these techniques, you can transform a general-purpose language model into a team of specialized virtual experts, ready to tackle complex challenges across any industry.

# References and Further Reading

- LLM Prompt Library (battle-tested prompt templates). https://abilzerian.github.io/LLM-Prompt-Library/
- AwesomeGPTSystemPrompts (GitHub collection). https://github.com/AntreasAntoniou/AwesomeGPTSystemPrompts
- Prompt Engineering Guide (Prompting Guide). https://www.promptingguide.ai/
- Prompt Creator: Prompt Engineering for AI (Prompt Creator). https://promptcreator.ai/