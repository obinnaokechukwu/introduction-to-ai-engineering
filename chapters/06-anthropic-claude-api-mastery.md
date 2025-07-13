# Chapter 6: Mastering the Anthropic Claude API

While OpenAI's GPT models are a fantastic entry point into the world of AI, a skilled developer knows that having multiple tools in their toolkit is essential. Anthropic's family of Claude models offers a powerful alternative, known for its strong reasoning capabilities, large context windows, and a focus on creating helpful, harmless, and honest AI systems.

This chapter is your complete guide to mastering the Claude API. We will cover everything from your first API call to building sophisticated applications that leverage Claude's unique strengths in document analysis, vision, and tool use. By understanding how to work with both OpenAI and Anthropic, you'll be able to choose the best model for any given task.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Make your first successful Claude API call and understand its core differences from OpenAI's API.
-   Craft powerful and detailed system prompts to precisely control Claude's behavior.
-   Analyze long documents and complex images using Claude's superior context and vision capabilities.
-   Implement Claude's "tool use" functionality to give your AI access to external functions.
-   Build a complete IoT fleet management system that uses visual diagnostics.
-   Compare the different Claude models and make informed decisions about when to use each.

## Your First Claude API Call: A New Dialect

Interacting with Claude will feel familiar yet distinct. The core concepts of roles and messages are the same, but the API structure has its own "dialect."

### The Absolute Minimum

First, ensure you have the `anthropic` library installed and an API key from the Anthropic console.

```bash
pip install anthropic
```

Here is the "Hello, World!" of the Claude API:

```python
import anthropic

# Initialize the client using an environment variable for your key
client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_API_KEY")

message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

# The response content is a list; we typically access the first text block.
print(message.content[0].text)
```

Let's break down the key differences from the OpenAI API:
-   The client is `anthropic.Anthropic()`.
-   The core method is `client.messages.create()`.
-   You **must** specify `max_tokens`.
-   The model name is specific to Claude (e.g., `claude-3-5-sonnet-20240620`).
-   The response text is located at `message.content[0].text`.

### A Reusable Claude Function

Let's formalize this into a secure and reusable function.

```python
import anthropic
import os
from dotenv import load_dotenv

# Load your API key securely from a .env file
load_dotenv()
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask_claude(prompt: str) -> str:
    """Sends a single prompt to Claude and returns the response."""
    try:
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"An error occurred with the Anthropic API: {e}"

# Let's test it
explanation = ask_claude("In simple terms, what makes the Claude 3.5 Sonnet model special?")
print(f"Claude's Response: {explanation}")
```

## Mastering Claude's System Prompts

One of Claude's standout features is its exceptional adherence to detailed **system prompts**. While OpenAI uses a `system` role within the `messages` list, Claude has a dedicated `system` parameter. This often leads to more reliable behavior and personality control.

```python
def create_claude_specialist(system_prompt: str, user_prompt: str) -> str:
    """Creates a specialized Claude assistant using a detailed system prompt."""
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=system_prompt,  # The dedicated system parameter
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    return message.content[0].text

# Define a detailed persona for an IoT diagnostic engineer
iot_expert_prompt = """
You are an expert IoT (Internet of Things) diagnostic engineer with 15 years of field experience. You are methodical, cautious, and practical.

When a user describes a problem, follow these steps precisely:
1.  Acknowledge the user's problem.
2.  Ask at least two clarifying questions to gather more specific information about the device, environment, or symptoms.
3.  Based on the initial description, provide 2-3 of the most likely root causes, ordered from most probable to least probable.
4.  For each potential cause, suggest a simple, safe diagnostic step the user can take.
5.  Conclude by advising the user to proceed with caution and report back with their findings.

Never suggest a solution without first asking questions. Prioritize safety and data integrity above all else.
"""

user_problem = "A dozen of my temperature sensors in the warehouse suddenly went offline."

# Get the expert diagnosis
diagnosis = create_claude_specialist(iot_expert_prompt, user_problem)
print(diagnosis)
```

The output from this prompt will be remarkably structured and will follow the prescribed steps, showcasing Claude's strength in adhering to complex instructions.

## Conversations and Long-Context Analysis

Claude's conversation flow is similar to OpenAI's—you build and manage the history. However, Claude models, particularly Sonnet and Opus, boast enormous context windows (200,000 tokens), making them ideal for analyzing entire documents or long conversation transcripts.

Let's create a chatbot that can "read" a document and then answer questions about it.

```python
class DocumentAnalyzerBot:
    def __init__(self, document_text: str):
        self.document = document_text
        self.system_prompt = f"""
You are an expert document analysis assistant. You have been provided with a document.
Answer the user's questions based *only* on the information contained within the following document.
If the answer is not in the document, say 'That information is not available in the provided document.'

--- DOCUMENT START ---
{self.document}
--- DOCUMENT END ---
"""
        self.conversation_history = []

    def ask(self, user_question: str) -> str:
        # Add the user's question to the conversation history
        self.conversation_history.append({"role": "user", "content": user_question})
        
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.conversation_history
        )
        
        claude_response = message.content[0].text
        # Add Claude's response to the history for follow-up questions
        self.conversation_history.append({"role": "assistant", "content": claude_response})
        
        return claude_response

# Simulate a long IoT device manual
device_manual = """
Product Manual: SmartSense T-1000 Temperature Sensor
Section 1: Installation. Mount the sensor away from direct heat sources.
Section 2: Battery. Uses two 3V CR2032 batteries. Expected life: 24 months.
Section 3: Connectivity. Connects via Zigbee 3.0. A flashing blue light indicates it is searching for a network. A solid blue light indicates it is connected.
Section 4: Troubleshooting. If readings are inaccurate, perform a recalibration cycle by holding the reset button for 10 seconds. If the device is unresponsive, replace the batteries.
"""

# Create an analyzer bot with the manual as its knowledge base
analyzer = DocumentAnalyzerBot(device_manual)

print("Bot:", analyzer.ask("How do I connect my new sensor to the network?"))
print("\nBot:", analyzer.ask("What kind of battery does it use?"))
print("\nBot:", analyzer.ask("What is the warranty period?"))
```

The bot correctly answers the first two questions based on the provided text and correctly states that the warranty information is not available.

## Multimodal Mastery: Vision and Tool Use

Claude 3.5 Sonnet is a state-of-the-art multimodal model, excelling at interpreting visual information and using tools.

### Vision: Analyzing Images

Claude can analyze images provided as base64-encoded strings. This is perfect for visual diagnostics.

```python
import base64

def analyze_iot_dashboard(image_path: str) -> str:
    """Sends an image of an IoT dashboard to Claude for analysis."""
    
    # Read the local image file and encode it
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
    prompt = "You are an IoT monitoring expert. Analyze this dashboard screenshot. Summarize the overall system status and identify any devices or metrics that require immediate attention."
    
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png", # Or image/jpeg, etc.
                        "data": image_data,
                    },
                },
            ],
        }]
    )
    return message.content[0].text

# To test this, you would need an image file named 'dashboard.png'
# For example: print(analyze_iot_dashboard('dashboard.png'))
```

### Tool Use: Claude's Function Calling

Claude's "tool use" is its equivalent of OpenAI's function calling. The workflow is very similar: you describe tools, the model requests to use one, you execute it, and you send the result back.

Let's build an IoT assistant with a tool to get live device data.

```python
import json

# This is our actual Python function (the "tool")
def get_live_device_status(device_id: str) -> str:
    """A dummy function to simulate fetching live device data."""
    print(f"--- TOOL EXECUTED: get_live_device_status for {device_id} ---")
    if device_id == "TEMP-042":
        return json.dumps({"status": "online", "temperature": "22.5°C", "battery": "88%"})
    return json.dumps({"status": "offline", "error": "Device not found"})

def iot_tool_bot(user_prompt: str):
    messages = [{"role": "user", "content": user_prompt}]
    
    # Define the tool for Claude
    tools = [{
        "name": "get_live_device_status",
        "description": "Get the live, up-to-the-second status of a specific IoT device by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string", "description": "The unique ID of the device, e.g., 'TEMP-042'"}},
            "required": ["device_id"],
        },
    }]

    # Step 1: First call to Claude
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # Check if Claude wants to use a tool
    if response.stop_reason != "tool_use":
        return response.content[0].text
        
    # Step 2: Claude wants to use a tool. Execute it.
    # The tool use request is typically the last item in the response content list
    tool_use = next(c for c in response.content if c.type == "tool_use")
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    print(f"Claude wants to call tool '{tool_name}' with arguments: {tool_input}")
    
    if tool_name == "get_live_device_status":
        tool_result = get_live_device_status(**tool_input)
    else:
        tool_result = "Error: Unknown tool."

    # Step 3: Send the tool's result back to Claude
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": tool_result,
        }]
    })

    final_response = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    return final_response.content[0].text

print(iot_tool_bot("What is the live status of device TEMP-042?"))
```

## When to Choose Claude vs. OpenAI

Now that you're familiar with both APIs, how do you choose?

| Use Case                                  | Recommended Model | Why?                                                                      |
| ----------------------------------------- | ----------------- | ------------------------------------------------------------------------- |
| **Long Document Analysis** (e.g., PDFs, reports) | **Claude**        | Superior large context window (200K tokens) handles entire documents easily.  |
| **Complex Reasoning & Strategy**          | **Claude**        | Often produces more nuanced, structured, and thoughtful responses.            |
| **Code Generation & Technical Tasks**     | **OpenAI (GPT-4o)** | Generally faster and highly proficient at writing and debugging code.          |
| **Cost-Sensitive, High-Volume Tasks**     | **OpenAI (GPT-4o mini)** | Tends to be more cost-effective for simple, high-frequency API calls.       |
| **Visual Analysis of Charts & Diagrams**  | **Claude**        | Excels at detailed interpretation of complex visual data like graphs and UIs. |
| **Strict Persona & Behavior Control**       | **Claude**        | Adherence to detailed system prompts is exceptionally reliable.           |
| **Fast, General-Purpose Chat**            | **Either**        | Both offer excellent models for general chat; test for your specific needs.  |

**A good strategy:** Use `gpt-4o-mini` for fast, cheap, general tasks, and `claude-3-5-sonnet` for tasks requiring deep reasoning, long-context analysis, or strict instruction following.

## Conclusion

You have now mastered the essentials of the Anthropic Claude API. You've seen how its architecture differs slightly from OpenAI's but enables powerful and unique capabilities.

You've learned to:
-   Interact with the Claude API using the `messages.create` endpoint.
-   Leverage detailed `system` prompts for precise behavioral control.
-   Build conversational bots that can analyze entire documents.
-   Use Claude's state-of-the-art vision capabilities for visual diagnostics.
-   Implement Claude's "tool use" to connect your AI to live data and functions.

By adding Claude to your developer toolkit, you've significantly expanded your ability to tackle a wider range of AI challenges. You can now choose the best model for the job, blending the strengths of different AI providers to create more powerful and effective applications.

# References and Further Reading

- Anthropic Claude API Documentation. https://docs.anthropic.com/
- Build with Claude (Anthropic Academy). https://www.anthropic.com/learn/build-with-claude
- Claude 3.5 Sonnet API Tutorial: Quick Start Guide. https://medium.com/ai-agent-insider/claude-3-5-sonnet-api-tutorial-quick-start-guide-anthropic-api-3f35ce56c59a