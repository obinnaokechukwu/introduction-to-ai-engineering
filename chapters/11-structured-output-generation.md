# Chapter 11: Mastering Structured Output

In software development, data is only useful if it's predictable. A web application expects a JSON object, a database expects a SQL query, and a configuration file expects a specific format like XML or YAML. While large language models are masters of generating creative, unstructured text, this flexibility becomes a liability when your application needs to programmatically process the AI's response.

This chapter is dedicated to the critical skill of forcing an AI's creative output into a rigid, reliable, and machine-readable structure. We will explore techniques that move beyond simple text generation to produce consistent JSON, XML, and even entire codebases that adhere to strict formatting and validation rules. Mastering structured output is the key to integrating AI as a reliable component in any automated system.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Consistently generate valid JSON, XML, and other structured data formats from AI models.
-   Integrate Pydantic models with your AI calls for automatic type safety and data validation.
-   Design prompts and error-handling loops that gracefully recover from malformed AI outputs.
-   Generate production-quality code that adheres to specific constraints and architectural patterns.
-   Build a complete, validated IoT device configuration generator from scratch.

## The Problem: The Unpredictability of Unstructured Text

Let's start by highlighting the problem we're trying to solve. Imagine asking an AI to analyze a sensor reading.

```python
import openai

client = openai.OpenAI()

def unstructured_analysis(sensor_data: str) -> str:
    """A standard prompt that results in unstructured, hard-to-parse text."""
    prompt = f"Analyze this IoT sensor data and tell me what's wrong and what to do: {sensor_data}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

sensor_reading = "device: TEMP-042, temp: 85°C, battery: 15%, signal: -92dBm"
analysis = unstructured_analysis(sensor_reading)
print("--- Unstructured Output ---")
print(analysis)
```
The response might be helpful for a human, but for a program, it's a nightmare. How would your code reliably extract the severity, the specific issues, and the recommended actions? You would have to write complex and brittle parsing logic using regular expressions. If the AI changes its wording slightly, your code breaks.

## The Solution: Prompting for Structure

The solution is to make the desired structure part of the prompt itself. We can explicitly instruct the AI to respond in a machine-readable format like JSON.

```python
import json

def structured_analysis(sensor_data: str) -> dict:
    """A prompt engineered to produce a specific JSON structure."""
    
    prompt = f"""
Analyze the following IoT sensor data.
Data: "{sensor_data}"

Respond with a JSON object that follows this exact structure:
{{
  "device_id": string,
  "health_status": "one of [Healthy, Warning, Critical]",
  "issues": [
    {{
      "component": "one of [Temperature, Battery, Connectivity]",
      "description": "A brief explanation of the issue."
    }}
  ],
  "recommended_action": "The single most important next step."
}}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        # The AI's response should be a string containing valid JSON
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response as JSON."}

structured_result = structured_analysis(sensor_reading)
print("\n--- Structured Output ---")
print(json.dumps(structured_result, indent=2))

# Now, your application can reliably access the data.
if structured_result.get("health_status") == "Critical":
    print("\nAction required:", structured_result.get("recommended_action"))
```
This is a huge improvement, but it's not foolproof. The model might still occasionally produce malformed JSON.

## Pattern 1: Guaranteed JSON with JSON Mode

To solve the problem of malformed JSON, modern models from OpenAI, Anthropic, and Google support a special **JSON Mode**. When enabled, the model is constrained to *only* output text that is a syntactically valid JSON object. This eliminates parsing errors entirely.

In the OpenAI API, you enable this by setting `response_format={"type": "json_object"}`.

```python
def guaranteed_json_analysis(device_data: str) -> dict:
    """Uses JSON Mode to guarantee a valid JSON response."""
    
    prompt = "Analyze this IoT data and respond with JSON containing keys: 'device_id', 'status', 'issues', and 'recommendation'."
    
    full_prompt = f"{prompt}\n\nData: {device_data}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}],
        # This is the key to enabling JSON Mode
        response_format={"type": "json_object"},
        temperature=0.1 # Lower temperature for more predictable structure
    )
    
    # No need for a try...except block for JSON parsing!
    return json.loads(response.choices[0].message.content)

json_output = guaranteed_json_analysis(sensor_reading)
print("--- Guaranteed JSON Output ---")
print(json.dumps(json_output, indent=2))
```
Using JSON Mode is a best practice for any application that needs to programmatically consume AI output.

## Pattern 2: Pydantic for Bulletproof Validation

While JSON Mode guarantees a valid *syntax*, it doesn't guarantee the correct *schema*. The AI might return a string where you expect a number, or omit a required field. This is where **Pydantic** comes in. Pydantic is a Python library for data validation that works seamlessly with modern IDEs and static analysis tools.

First, define your desired data structure as a Pydantic `BaseModel`. This class serves as both your schema definition and your validator.

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class DeviceIssue(BaseModel):
    component: Literal["Temperature", "Battery", "Connectivity"]
    description: str = Field(..., min_length=10) # `...` means the field is required

class DeviceAnalysis(BaseModel):
    device_id: str
    health_status: Literal["Healthy", "Warning", "Critical"]
    issues: List[DeviceIssue]
    health_score: int = Field(..., ge=0, le=100) # Must be an integer between 0 and 100
```
Now, we can integrate this into our AI call. We first generate the JSON schema from our Pydantic model to include in the prompt, then we use the model to parse and validate the AI's response.

```python
def pydantic_validated_analysis(sensor_data: str) -> DeviceAnalysis:
    """Generates and validates a response using a Pydantic model."""
    
    # Pydantic can generate a JSON schema from your model
    json_schema = DeviceAnalysis.model_json_schema()
    
    prompt = f"""
Analyze the provided IoT data. You MUST respond with a JSON object
that strictly follows this JSON Schema:
{json.dumps(json_schema, indent=2)}

Data to analyze: "{sensor_data}"
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    ).choices[0].message.content
    
    try:
        # Pydantic will parse the JSON and validate it against your model.
        # If validation fails, it raises a ValidationError.
        validated_data = DeviceAnalysis.model_validate_json(response)
        return validated_data
    except Exception as e:
        print(f"Pydantic validation failed: {e}")
        print(f"Raw AI response was: {response}")
        # In a real app, you could implement a retry loop here to ask the AI to fix its own output.
        return None

validated_output = pydantic_validated_analysis(sensor_reading)
if validated_output:
    print("\n--- Pydantic-Validated Output ---")
    # .model_dump_json() gives a nicely formatted JSON string
    print(validated_output.model_dump_json(indent=2))
    
    # You can now access data with type safety and autocompletion
    print(f"\nHealth Score: {validated_output.health_score}")
```
This combination of JSON Mode and Pydantic validation is the gold standard for production-grade structured output. It ensures that the data your application receives is not only syntactically correct but also semantically valid according to your business logic.

## Pattern 3: Constrained Code Generation

Generating structured data is one thing; generating functional, correct, and secure code is another. When asking an AI to write code, you must provide strong constraints and patterns.

### Generating a Device Driver

Let's ask the AI to generate a Python class for a specific type of IoT device, but with strict requirements.

```python
def generate_device_driver(device_spec: dict) -> str:
    """Generates a Python device driver class based on a specification."""
    
    prompt = f"""
Generate a complete, production-ready Python class for an IoT device based on the following specification.

**Specification:**
{json.dumps(device_spec, indent=2)}

**Mandatory Requirements:**
1.  The class must be named `ModbusTempSensor`.
2.  It must include full type hints for all method arguments and return values.
3.  It must include comprehensive docstrings for the class and all public methods.
4.  It must include `try...except` blocks for all I/O operations to handle potential errors gracefully.
5.  It must use the `logging` module to log informational messages and errors.
6.  The code must be PEP 8 compliant.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o", # Use a more powerful model for better code generation
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# Define the specification for our desired device driver
sensor_spec = {
    "device_type": "Modbus Temperature Sensor",
    "protocol": "Modbus RTU",
    "connection": "Serial (RS-485)",
    "registers": {
        "temperature": {"address": 4001, "type": "int16", "scale_factor": 0.1},
        "humidity": {"address": 4002, "type": "uint16", "scale_factor": 0.1}
    }
}

driver_code = generate_device_driver(sensor_spec)
print("--- Generated Device Driver Code ---")
print(driver_code)
```
By providing a clear list of non-negotiable requirements, you guide the AI to produce code that aligns with your project's standards, rather than a simplistic and unusable snippet.

### Template-Based Code Generation

An even more robust method is to provide a template and ask the AI to fill in the blanks. This gives you maximum control over the final structure.

```python
def generate_from_template(template: str, parameters: dict) -> str:
    """Fills in a code template using AI."""
    
    prompt = f"""
Fill in the following Python code template using the provided parameters.
Only replace the placeholders in curly braces (e.g., {{CLASS_NAME}}).
Do not change the structure of the template.

**Template:**
```python
{template}
```

**Parameters:**
{json.dumps(parameters, indent=2)}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    ).choices[0].message.content
    
    # Clean up the response to get just the code
    if "```python" in response:
        response = response.split("```python")[1]
    if "```" in response:
        response = response.split("```")[0]
        
    return response.strip()

# A generic template for an API endpoint in FastAPI
api_endpoint_template = """
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/{endpoint_prefix}", tags=["{tag_name}"])

class {model_name}Response(BaseModel):
    id: str
    status: str

@router.get("/{endpoint_path}/{{item_id}}", response_model={model_name}Response)
async def get_{singular_name}(item_id: str):
    \"\"\" {endpoint_description} \"\"\"
    # --- Start AI-generated logic ---
    # {business_logic_placeholder}
    # --- End AI-generated logic ---
    raise HTTPException(status_code=404, detail="Item not found")
"""

# Parameters to fill in the template
params = {
    "endpoint_prefix": "devices",
    "tag_name": "Devices",
    "model_name": "Device",
    "endpoint_path": "status",
    "singular_name": "device_status",
    "endpoint_description": "Gets the current status of a specific IoT device.",
    "business_logic_placeholder": "Find device in database and return its status."
}

generated_code = generate_from_template(api_endpoint_template, params)
print("\n--- Template-Generated API Endpoint ---")
print(generated_code)
```

## Practical Project: A Validated IoT Configuration Generator

Let's combine these patterns to build a powerful tool. Our goal is to create a system that takes a high-level description of an IoT deployment and generates a complete, validated JSON configuration file for the devices.

```python
from typing import Literal

# 1. Define the complete, desired output structure with Pydantic
class SensorConfig(BaseModel):
    sensor_type: Literal["temperature", "humidity", "pressure"]
    reading_interval_seconds: int = Field(..., ge=10) # Must be at least 10 seconds

class NetworkConfig(BaseModel):
    connection_type: Literal["WiFi", "Cellular", "LoRaWAN"]
    wifi_ssid: str | None = None
    
class DeviceConfiguration(BaseModel):
    device_id_prefix: str
    location: str
    network: NetworkConfig
    sensors: List[SensorConfig]
    firmware_version: str = "1.0.0"

class ConfigurationGenerator:
    def __init__(self):
        self.client = openai.OpenAI()

    def generate(self, requirements: str) -> DeviceConfiguration:
        # 2. Get the JSON schema from our Pydantic model
        schema = DeviceConfiguration.model_json_schema()
        
        prompt = f"""
Generate a complete IoT device configuration file based on these high-level requirements.
You MUST respond with a JSON object that strictly validates against the provided JSON Schema.

**JSON Schema to follow:**
{json.dumps(schema, indent=2)}

**User Requirements:**
"{requirements}"
"""
        # 3. Use JSON mode to get a syntactically valid response
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        ).choices[0].message.content
        
        # 4. Use Pydantic to parse and validate the response
        try:
            validated_config = DeviceConfiguration.model_validate_json(response)
            return validated_config
        except Exception as e:
            print(f"Initial generation failed validation: {e}")
            # 5. Implement a retry loop to ask the AI to fix its own mistake
            return self._fix_and_retry(prompt, response, str(e))

    def _fix_and_retry(self, original_prompt, failed_response, error_message):
        print("Attempting to self-correct the generated JSON...")
        fix_prompt = f"""
The JSON you previously generated failed validation with the following error:
Error: "{error_message}"

Original failed JSON:
{failed_response}

Please review the original prompt, the JSON schema, and the error message.
Then, generate a new, corrected JSON object that fixes the error and perfectly matches the schema.

Original Prompt:
{original_prompt}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": fix_prompt}],
            response_format={"type": "json_object"}
        ).choices[0].message.content
        
        # Try validating one last time
        return DeviceConfiguration.model_validate_json(response)

# --- Demo of the complete system ---
generator = ConfigurationGenerator()
user_req = "I need to deploy temperature and humidity sensors in our cold storage warehouse. They should report every 5 minutes and connect via our 'Warehouse-IoT' WiFi network."

final_config = generator.generate(user_req)

print("--- Final Validated Configuration ---")
print(final_config.model_dump_json(indent=2))
```
This system demonstrates a robust, production-ready workflow for structured output generation. It clearly defines the desired output, uses JSON mode for syntactic correctness, validates the data with Pydantic for semantic correctness, and even includes a self-correction loop to handle errors gracefully.

## Conclusion

Structured output is the bridge between the creative, probabilistic world of LLMs and the deterministic, logical world of software applications. By mastering these patterns, you can build reliable data pipelines, automated code generators, and robust configuration systems powered by AI.

Remember these key practices:
-   **Be Explicit:** Tell the AI the exact format you want. Provide a schema or examples.
-   **Use JSON Mode:** Guarantee syntactic validity and eliminate parsing errors.
-   **Validate with Pydantic:** Enforce your data contract with type hints, constraints, and business rules.
-   **Constrain Code Generation:** Use templates and clear requirements to get production-quality code.
-   **Implement Self-Correction:** Build retry loops that ask the AI to fix its own errors.

You are now equipped to build AI systems that don't just talk, but speak the precise, structured language that your applications understand.

# References and Further Reading

- Structured Output in Large Language Models (LLMs) (Medium). https://mehmetozkaya.medium.com/structured-output-in-large-language-models-llms-88e6f8602e25
- Structured Output Generation in LLMs: Techniques, Comparisons, and Applications (Zero Point Labs). https://zeropointlabs.ai/structured-output-generation-in-llms-techniques-comparisons-and-applications/
- 4. Structured Output (Taming LLMs). https://www.tamingllms.com/notebooks/structured_output.html
- Structured outputs in LLMs: Definition, techniques, applications, benefits (LeewayHertz). https://www.leewayhertz.com/structured-outputs-in-llms/