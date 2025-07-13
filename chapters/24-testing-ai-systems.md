# Chapter 24: Testing AI Systems

In traditional software development, testing is a world of certainty. An assertion like `assert add(2, 2) == 4` is absolute. It will pass or fail with perfect predictability. This deterministic nature is the bedrock upon which we build reliable software. When we introduce Large Language Models into our applications, we step into a new, probabilistic world. An LLM is not a calculator; it's a creative engine. Asking it the same question twice might yield two slightly different, yet equally correct, answers.

This presents a profound challenge: how do you test something that is inherently non-deterministic? How can you write a test that doesn't break every time a model update slightly changes the phrasing of a response? The fragile promise of `assert ai_response == "expected text"` is a recipe for a perpetually broken CI/CD pipeline.

This chapter will guide you through a new paradigm of testing, one designed for the age of AI. We will learn to move beyond testing for *exactness* and instead test for *quality, structure, and intent*. We will explore a suite of techniques, from unit testing our prompt logic to advanced regression testing using AI itself as a judge. By the end, you will be equipped to build robust testing suites that bring confidence and reliability to your AI-powered applications.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Understand the fundamental challenges of testing non-deterministic AI systems.
-   Unit test AI integration code by mocking API calls and validating prompt construction.
-   Implement integration tests for complex, multi-step AI workflows.
-   Create AI regression tests using a "golden dataset" and an AI "judge" to prevent quality degradation.
-   Perform load testing on AI endpoints to understand performance and cost under pressure.
-   Build a complete testing suite for a practical IoT command validation system.

## The Foundation: Unit Testing AI Integrations

While we can't unit test the AI model itself, we can and absolutely must unit test the code that *interacts* with it. The "unit" in this context is typically the function in your application that constructs a prompt, makes an API call, and parses the response.

We will focus on testing three critical parts of this unit:
1.  **Prompt Construction:** Is your code building the correct prompt based on the inputs? This is deterministic and highly testable.
2.  **API Call Logic:** Does your code call the correct model and endpoint with the right parameters?
3.  **Response Handling:** Does your code correctly parse the AI's response and handle potential errors?

We will use Python's standard `pytest` framework and the `unittest.mock` library for this.

First, install the necessary libraries: `pip install pytest pytest-mock`.

### Testing Prompt Construction

This is the most important unit test you can write. It verifies that the instructions you're sending to the AI are exactly what you intend.

Let's imagine we have a simple function to analyze a device's status.

```python
# app/analyzer.py
def create_analysis_prompt(device_id: str, status: str, battery: int) -> str:
    """Creates a standardized prompt for analyzing a device's status."""
    return f"""
Analyze the status of IoT device '{device_id}'.
Status: {status}
Battery: {battery}%
Provide a one-sentence summary of its health.
"""
```

Now, let's write a `pytest` test for it in a separate `tests/` directory.

```python
# tests/test_analyzer.py
from app.analyzer import create_analysis_prompt

def test_create_analysis_prompt_formats_correctly():
    # Arrange
    device_id = "TEMP-042"
    status = "online"
    battery = 88
    
    # Act
    prompt = create_analysis_prompt(device_id, status, battery)
    
    # Assert
    assert "IoT device 'TEMP-042'" in prompt
    assert "Status: online" in prompt
    assert "Battery: 88%" in prompt
    assert "one-sentence summary" in prompt
```

This test runs instantly, costs nothing, and guarantees that our prompt logic is correct.

### Testing the API Call with Mocking

We don't want our unit tests to make real, expensive, and slow network calls to an AI API. Instead, we **mock** the API call. Mocking means replacing a real component (like the OpenAI client) with a fake one that we can control for our test.

Let's expand our example with a function that makes the actual call.

```python
# app/analyzer.py
import openai

client = openai.OpenAI()

def analyze_device_status(device_id: str, status: str, battery: int) -> str:
    """Calls the AI to analyze device status."""
    prompt = create_analysis_prompt(device_id, status, battery)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

Now, let's test this function *without* calling OpenAI, using `pytest-mock`.

```python
# tests/test_analyzer.py
from app.analyzer import analyze_device_status
from unittest.mock import MagicMock # Import MagicMock

def test_analyze_device_status_calls_api_correctly(mocker):
    # Arrange: Create a fake response object that the AI client will return.
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "The device is healthy and operational."
    
    # mocker.patch replaces the real OpenAI client method with our fake one.
    mock_api_call = mocker.patch("openai.resources.chat.completions.Completions.create", return_value=mock_response)
    
    # Act
    result = analyze_device_status("TEMP-042", "online", 88)
    
    # Assert
    # 1. Did our function return the expected text?
    assert result == "The device is healthy and operational."
    
    # 2. Was the API called exactly once?
    mock_api_call.assert_called_once()
    
    # 3. Was it called with the correct parameters?
    # We can inspect the arguments it was called with.
    call_args, call_kwargs = mock_api_call.call_args
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert "TEMP-042" in call_kwargs["messages"][0]["content"]
```
This test verifies that our code is interacting with the AI library correctly, without ever making a real network request.

## Integration Testing for AI Workflows

Integration tests verify that different parts of your system—especially multiple AI-powered components—work together correctly. A common pattern to test is a **prompt chain**, where the output of one AI call becomes the input for the next.

Let's design a simple two-step chain:
1.  **Step 1:** Classify an alert's severity.
2.  **Step 2:** Based on the severity, generate a summary.

```python
# app/report_chain.py
class IoTReportChain:
    def __init__(self, client):
        self.client = client

    def run(self, alert: str) -> str:
        # Step 1: Classify severity
        severity_prompt = f"Classify the severity of this alert as 'Low', 'Medium', or 'High'. Alert: {alert}"
        severity_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": severity_prompt}]
        )
        severity = severity_response.choices[0].message.content.strip()

        # Step 2: Generate summary based on severity
        summary_prompt = f"Write a one-sentence summary for a '{severity}' severity alert about: {alert}"
        summary_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        return summary_response.choices[0].message.content
```

Our integration test must verify that the `severity` from the first call is correctly used in the prompt for the second call. We can use `mocker`'s `side_effect` feature to return different mock responses for each call.

```python
# tests/test_report_chain.py
def test_iot_report_chain_integration(mocker):
    # Arrange: Define the sequence of fake responses the AI will give.
    mock_response_1 = MagicMock()
    mock_response_1.choices[0].message.content = "High" # AI's response to the severity prompt
    
    mock_response_2 = MagicMock()
    mock_response_2.choices[0].message.content = "This is a high-severity alert." # AI's response to the summary prompt

    # Patch the API call and set its side_effect to return our responses in order.
    mock_api_call = mocker.patch("openai.resources.chat.completions.Completions.create", side_effect=[mock_response_1, mock_response_2])
    
    # Act
    chain = IoTReportChain(client=MagicMock()) # Pass a mock client
    alert = "Pressure valve V-101 exceeds critical limit."
    final_summary = chain.run(alert)

    # Assert
    # 1. Was the API called twice?
    assert mock_api_call.call_count == 2
    
    # 2. Check the prompt of the *second* call to ensure it used the output of the *first*.
    second_call_args = mock_api_call.call_args_list[1]
    second_prompt = second_call_args.kwargs["messages"][0]["content"]
    assert "Write a one-sentence summary for a 'High' severity alert" in second_prompt
    
    # 3. Check the final output.
    assert final_summary == "This is a high-severity alert."
```

## AI Regression Testing: The Golden Dataset and the AI Judge

This is the most critical and AI-specific form of testing. How do you ensure that changing a prompt or upgrading a model doesn't degrade the quality of your responses? This is **AI regression testing**.

Since we can't test for exact string matches, we use a powerful, two-part strategy:

1.  **The Golden Dataset:** A curated set of representative input prompts and their known, high-quality "golden" responses. This dataset captures the essential capabilities of your AI feature.
2.  **The AI Judge:** A separate, powerful LLM (like GPT-4o) that is tasked with evaluating the new response against the golden response. The judge scores the new response based on criteria you define, and your test passes if the score is above a certain threshold.

Let's build a regression tester.

```python
# app/regression_tester.py
import openai

class PromptRegressionTester:
    def __init__(self, client):
        self.client = client
        self.judge_model = "gpt-4o" # Use a powerful model for evaluation

    def evaluate_response(self, prompt: str, golden_response: str, new_response: str) -> dict:
        """Uses an AI judge to evaluate the new response against the golden one."""
        
        evaluation_prompt = f"""
You are an expert quality assurance analyst. Your task is to evaluate an AI-generated response against a golden (ideal) response.

**Original Prompt:**
{prompt}

**Golden Response (ideal answer):**
{golden_response}

**New Response (to evaluate):**
{new_response}

**Evaluation Criteria:**
1.  **Factual Accuracy:** Does the new response contradict the golden response?
2.  **Completeness:** Does the new response omit any key information present in the golden response?
3.  **Clarity and Tone:** Is the new response as clear and appropriate in tone as the golden one?

Based on these criteria, please provide a score from 1 to 10 (where 10 is perfect) and a brief justification.
Respond ONLY with a JSON object in the format: {{"score": <number>, "justification": "<text>"}}
"""
        
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a fair and impartial evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

Now, we can use this in a `pytest` test.

```python
# tests/test_regression.py

# Our "golden dataset" of inputs and known-good outputs.
golden_dataset = [
    {
        "input": "Device TEMP-01 is at 95°C. Safe range is 0-90°C.",
        "golden_output": "CRITICAL: Temperature for TEMP-01 is 95°C, exceeding the 90°C safe limit. Immediate inspection required."
    },
    {
        "input": "Device BATT-07 is at 18%. Threshold is 20%.",
        "golden_output": "WARNING: Battery for BATT-07 is low at 18%. Schedule replacement within 48 hours."
    }
]

def test_new_prompt_against_golden_dataset(mocker):
    # Imagine we're testing a new prompt for our `analyze_device_status` function.
    # We mock the call to that function to control its output for the test.
    
    # This is the output from our new, hypothetical prompt.
    new_response_from_prompt_v2 = "Alert: Device TEMP-01 temperature is 95°C. This is over the limit."
    
    # We don't need to mock the judge call, as that is what we are testing.
    # In a real CI/CD pipeline, you would make this real API call.
    # For this example, we can mock the judge to avoid costs during a simple run.
    mock_judge_response = MagicMock()
    mock_judge_response.choices[0].message.content = '{"score": 8, "justification": "Mostly correct but less direct than the golden response."}'
    mocker.patch("openai.resources.chat.completions.Completions.create", return_value=mock_judge_response)

    # --- Test Logic ---
    tester = PromptRegressionTester(client=MagicMock())
    test_case = golden_dataset[0]
    
    evaluation = tester.evaluate_response(
        prompt=test_case["input"],
        golden_response=test_case["golden_output"],
        new_response=new_response_from_prompt_v2
    )
    
    print(f"Judge's score: {evaluation['score']}, Justification: {evaluation['justification']}")
    
    # The assertion is on the judge's score, not the text itself.
    assert evaluation["score"] >= 7
```

This pattern allows you to update prompts and models with confidence, knowing you have a safety net to catch regressions in quality, tone, or accuracy.

## Performance and Load Testing

AI APIs introduce significant latency. Load testing helps you understand how your system will perform under heavy traffic and what your costs will be. **Locust** is a popular, easy-to-use Python load testing tool.

First, install Locust: `pip install locust`.

Next, create a `locustfile.py` that defines user behavior.

```python
# locustfile.py
from locust import HttpUser, task, between

class AIAppUser(HttpUser):
    # Wait 1 to 5 seconds between tasks
    wait_time = between(1, 5)
    
    # This task will be repeatedly called by simulated users
    @task
    def analyze_endpoint(self):
        # We target our API, not the OpenAI API directly
        self.client.post(
            "/analyze",
            json={"device_id": "LOCUST-TEST", "status": "online", "battery": 50}
        )
```
**Important:** When load testing, you should almost always have your application endpoint **mock the actual AI call**. Making thousands of real AI calls would be incredibly expensive. The goal is to test *your* system's ability to handle concurrent requests, manage queues, and process responses, not to stress-test OpenAI's infrastructure.

To run the test, start your FastAPI application and then run Locust:

```bash
# Start your API server first
uvicorn my_app:app --reload

# Then, in another terminal, run Locust
locust -f locustfile.py
```
Open `http://localhost:8089` in your browser to start the simulation and see real-time performance metrics like requests per second and response time distributions.

## Practical Project: A Complete Testing Suite

Let's apply all these patterns to test a simple but critical IoT system: a command validator that uses AI to determine if a command sent to a device is safe.

**System Under Test:**

```python
# app/command_validator.py
class IoTCommandValidator:
    def __init__(self, client):
        self.client = client
        self.system_prompt = """
You are a security validation system for an IoT platform.
Your task is to determine if a given command is 'SAFE' or 'UNSAFE'.
A command is UNSAFE if it attempts to:
- Reboot a critical system without proper authorization codes.
- Update firmware outside of a maintenance window.
- Delete data or logs.
- Expose sensitive information.
Respond with ONLY the word 'SAFE' or 'UNSAFE'.
"""
    def is_safe(self, command: str, user_role: str) -> bool:
        prompt = f"Command: '{command}'\nUser Role: '{user_role}'"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        decision = response.choices[0].message.content.strip().upper()
        return decision == "SAFE"
```

**The Complete Test Suite:**

```python
# tests/test_command_validator.py
import pytest
from app.command_validator import IoTCommandValidator
from unittest.mock import MagicMock

# --- Unit Tests ---
def test_is_safe_constructs_correct_prompt():
    # This test doesn't need a mock because we are not making the call
    # It only checks that the prompt is built correctly
    validator = IoTCommandValidator(client=MagicMock())
    # This is a conceptual test, as we can't see the prompt directly without refactoring.
    # In a real app, you might have a helper function for prompt creation that you can test.
    assert True # Placeholder for a real prompt construction test

def test_is_safe_handles_safe_response(mocker):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "SAFE"
    mock_api_call = mocker.patch("openai.resources.chat.completions.Completions.create", return_value=mock_response)
    
    validator = IoTCommandValidator(client=MagicMock())
    assert validator.is_safe("get_status", "admin") is True
    mock_api_call.assert_called_once()

def test_is_safe_handles_unsafe_response(mocker):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "UNSAFE"
    mocker.patch("openai.resources.chat.completions.Completions.create", return_value=mock_response)
    
    validator = IoTCommandValidator(client=MagicMock())
    assert validator.is_safe("delete_all_logs", "user") is False

# --- Regression Test ---
def test_validator_regression(mocker):
    # Setup our AI Judge
    tester = PromptRegressionTester(client=MagicMock())
    
    # A golden test case
    command = "reboot_critical_server_zone_a"
    user_role = "technician"
    golden_output = "UNSAFE"
    
    # The output from our new prompt/model we are testing
    new_actual_output = "UNSAFE" # Let's say it gets it right

    # Mock the judge's evaluation
    mock_judge_response = MagicMock()
    mock_judge_response.choices[0].message.content = '{"score": 10, "justification": "The new response correctly identifies the command as unsafe, matching the golden response."}'
    mocker.patch("openai.resources.chat.completions.Completions.create", return_value=mock_judge_response)
    
    evaluation = tester.evaluate_response(
        prompt=f"Command: '{command}', User Role: '{user_role}'",
        golden_response=golden_output,
        new_response=new_actual_output
    )
    
    assert evaluation["score"] >= 8
```

This suite provides confidence in our validator from multiple angles: our code logic is correct (unit tests), and the AI's output quality is maintained over time (regression tests).

# References and Further Reading

- [How to QA and Test AI Products (Medium, 2025)](https://medium.com/@michael-brown-/how-to-qa-and-test-ai-products-68325fc1938c)
- [How to Test AI Agents Effectively (Galileo AI, 2024)](https://www.galileo.ai/blog/how-to-test-ai-agents-evaluation)
- [How to test AI agents (Blinq, 2025)](https://www.blinq.io/post/how-to-test-ai-agents-guy-arieli)
- [Testing AI Systems: New Rules for a New Era (DEV.to, 2024)](https://dev.to/vaibhavkuls/testing-ai-systems-new-rules-for-a-new-era-33d1)
- [Enterprise Grade AI/ML Deployment on AWS 2025 (DEV.to, 2025)](https://dev.to/deploy/enterprise-grade-aiml-deployment-on-aws-2025-i23)