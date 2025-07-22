# Chapter 10: Advanced Prompting Strategies

In the last chapter, we mastered the fundamentals of prompt engineering. We learned to be specific, provide context, and use patterns like few-shot and chain-of-thought. These techniques are the bedrock of effective AI interaction and will serve you well for 80% of use cases.

But what about the other 20%? What happens when a problem is so complex, so ambiguous, or so high-stakes that a single, direct prompt is insufficient? This is where advanced prompting strategies come into play. These are not just different ways of asking a question; they are structured, multi-step reasoning frameworks that guide the AI to tackle complexity in a more robust and reliable way. This chapter is your introduction to thinking like an AI architect, designing not just single prompts, but entire reasoning workflows.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Implement **Self-Consistency** to improve the reliability of AI-generated answers.
-   Use **Tree-of-Thought (ToT)** prompting to explore multiple reasoning paths for complex problems.
-   Apply the **ReAct (Reasoning and Acting)** pattern to build agents that can dynamically gather information.
-   Build sophisticated **Prompt Chains** that break down a complex task into a sequence of specialized prompts.

## The Limits of Simple Prompts

Basic prompting is powerful, but it has its limits. When faced with a highly complex problem, a single prompt can lead to inconsistent or shallow analysis.

Consider this complex IoT system failure:

```python
complex_problem = """
A smart factory's assembly line has stopped. Key symptoms include:
- Conveyor belt motors in Zone C are offline.
- Vibration sensors in the adjacent Zone B are showing erratic, high-amplitude readings.
- Network latency for all devices in Zone C has increased by 500ms.
- The issue began an hour after the morning shift started.
- No new firmware has been deployed recently.
"""

# A basic prompt might give a plausible but incomplete answer,
# potentially missing the root cause that connects all symptoms.
```
A single pass from an LLM might latch onto one symptom (e.g., "network latency") and propose a solution without considering how it connects to the others. Advanced strategies are designed to overcome this by forcing a more rigorous, multi-faceted analysis.

## Strategy 1: The Self-Consistency Method

**The Idea:** If you ask one expert their opinion, you get one answer. If you ask five experts, you can find a consensus. The self-consistency method applies this logic to AI. Instead of running a prompt once, you run it multiple times with a higher `temperature` (to encourage varied responses) and then find the most common answer.

This technique is incredibly effective at improving accuracy on tasks with a single correct answer, such as arithmetic or logical reasoning.

### Implementing Self-Consistency

Let's apply this to find the root cause of a device failure.

```python
import openai
import json
from collections import Counter
from typing import List, Dict, Any

client = openai.OpenAI()

def self_consistency_diagnosis(problem: str, num_runs: int = 5) -> str:
    """Runs a diagnosis prompt multiple times to find the most consistent root cause."""
    
    prompt = f"""
Analyze the following IoT issue and state the single most likely root cause.

Problem: {problem}

Most Likely Root Cause:
"""
    
    responses = []
    print(f"Running diagnosis {num_runs} times to find consensus...")

    for i in range(num_runs):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 # Higher temperature encourages diverse reasoning paths
        ).choices[0].message.content
        
        # We only care about the stated root cause for this simple example
        root_cause = response.strip()
        print(f"  Run {i+1} Diagnosis: {root_cause}")
        responses.append(root_cause)

    # Find the most common response using a Counter
    if not responses:
        return "Failed to get any valid responses."
        
    most_common_cause, count = Counter(responses).most_common(1)[0]
    
    return f"Consensus Diagnosis (agreed upon in {count}/{num_runs} runs): {most_common_cause}"

# A problem where a single prompt might give different answers
device_issue = "A smart thermostat is reading 10°F higher than the actual room temperature. It was installed last week and has the latest firmware."

print(self_consistency_diagnosis(device_issue))
```
While one run might suggest "faulty sensor," another might suggest "improper placement near a heat source." By running it multiple times, the most plausible answer ("improper placement" for a new installation) is likely to emerge as the consensus.

**When to use it:** For high-stakes questions where accuracy is critical and there is likely a single best answer. It increases API costs but significantly improves reliability.

## Strategy 2: Tree-of-Thought (ToT) Prompting

**The Idea:** Complex problems are rarely solved in a straight line. Experts often explore several possible paths of reasoning, abandoning those that lead to dead ends and pursuing the most promising ones. Tree-of-Thought prompting simulates this process.

The workflow is:
1.  **Generate Thoughts:** Ask the AI to brainstorm several different initial approaches or lines of reasoning to solve the problem.
2.  **Develop Thoughts:** For each initial thought, ask the AI to elaborate on it, exploring its implications and required next steps.
3.  **Evaluate Thoughts:** Ask the AI to act as an expert critic, evaluating the developed thoughts to determine which one is the most promising.

### Implementing a ToT Workflow

```python
class TreeOfThoughtSolver:
    def __init__(self):
        self.client = openai.OpenAI()

    def solve(self, problem: str, num_paths: int = 3) -> str:
        # 1. Generate initial thoughts (the "branches" of the tree)
        print("--- Stage 1: Generating initial reasoning paths ---")
        initial_thoughts = self._generate_initial_thoughts(problem, num_paths)
        print("Initial Paths:", initial_thoughts)

        # 2. Develop each thought into a more detailed plan
        print("\n--- Stage 2: Developing each path ---")
        developed_paths = []
        for thought in initial_thoughts:
            developed_paths.append(self._develop_path(problem, thought))

        # 3. Evaluate the developed paths to find the best one
        print("\n--- Stage 3: Evaluating paths and synthesizing a final answer ---")
        return self._evaluate_and_synthesize(problem, developed_paths)

    def _generate_initial_thoughts(self, problem: str, num_thoughts: int) -> list[str]:
        prompt = f"""
Problem: "{problem}"
Generate {num_thoughts} distinct, high-level approaches to diagnose this problem. Each approach should be a single sentence.
"""
        response = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        return [line.strip() for line in response.split('\n') if line.strip()]

    def _develop_path(self, problem: str, thought: str) -> str:
        prompt = f"""
Problem: "{problem}"
Reasoning Path: "{thought}"
Elaborate on this reasoning path. What are the specific steps you would take to investigate this? What are the pros and cons of this approach?
"""
        return self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

    def _evaluate_and_synthesize(self, problem: str, developed_paths: list[str]) -> str:
        paths_text = "\n\n".join(f"Path {i+1}:\n{path}" for i, path in enumerate(developed_paths))
        prompt = f"""
You have explored several reasoning paths to solve this problem: "{problem}"

Here are the explored paths:
{paths_text}

Now, act as an expert reviewer. Evaluate the paths and synthesize the best elements into a single, optimal action plan.
"""
        return self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

# Let's use ToT to solve our complex factory problem
tot_solver = TreeOfThoughtSolver()
solution = tot_solver.solve(complex_problem)
print("\n--- Final ToT Solution ---")
print(solution)
```

**When to use it:** For complex, open-ended problems with no obvious solution. It helps avoid premature conclusions by forcing the model to consider multiple possibilities.

## Strategy 3: ReAct (Reasoning and Acting)

**The Idea:** Many problems can't be solved with existing information alone. You need to gather more data. The ReAct pattern creates a dynamic loop where the AI **reasons** about what it needs to know and then chooses an **action** (like calling a tool) to find that information.

This turns the LLM from a static knowledge base into an active agent that can interact with its environment.

### Implementing a ReAct Loop

```python
class ReActAgent:
    def __init__(self, available_tools: dict):
        self.client = openai.OpenAI()
        self.tools = available_tools

    def run(self, problem: str, max_steps: int = 5):
        thought_history = ""
        for i in range(max_steps):
            print(f"\n--- ReAct Step {i+1} ---")
            
            # 1. Reason about the next action
            prompt = f"""
You are an IoT diagnostic agent. Solve the problem by reasoning and then taking an action.
Problem: "{problem}"
Thought History:
{thought_history}

Your available tools are: {list(self.tools.keys())}

Thought: Analyze the current situation and decide on the next best action to take.
Action: [Choose one tool from the list]
"""
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            
            # Parse the thought and action
            thought = response.split("Action:")[0].replace("Thought:", "").strip()
            action = response.split("Action:")[1].strip()
            print(f"Thought: {thought}")
            print(f"Action: {action}")
            
            # 2. Execute the chosen action
            if action in self.tools:
                tool_result = self.tools[action]() # In a real app, you'd pass arguments
                print(f"Observation: {tool_result}")
                
                # Update the history for the next loop
                thought_history += f"Step {i+1}:\nThought: {thought}\nAction: {action}\nObservation: {tool_result}\n"
                
                # A simple check for a final answer
                if "root cause is" in tool_result.lower():
                    print("\n--- Diagnosis Complete ---")
                    return tool_result
            else:
                print("Action not recognized. Ending loop.")
                break
        
        return "Could not determine root cause within the step limit."

# --- Define some dummy tools for the demo ---
def check_network_logs():
    return "Network logs show high latency in Zone C, starting at 09:05 AM."
def check_motor_status():
    return "All motors in Zone C are reporting 'offline' via the API."
def review_maintenance_schedule():
    return "No recent maintenance was scheduled for Zone C. However, a power grid switchover occurred at 09:02 AM."

# --- Run the ReAct agent ---
tools = {
    "check_network_logs": check_network_logs,
    "check_motor_status": check_motor_status,
    "review_maintenance_schedule": review_maintenance_schedule,
}

react_agent = ReActAgent(tools)
react_agent.run(complex_problem)
```

**When to use it:** For dynamic problems where the AI needs to gather information incrementally to solve a problem. This is the foundation of autonomous agents.



## Conclusion

You have now moved beyond basic prompting and into the realm of advanced AI reasoning frameworks. These strategies are the tools that allow you to tackle ambiguity, complexity, and high-stakes problems with a greater degree of confidence and reliability.

-   **Self-Consistency:** Use for accuracy and to reduce the randomness of single-shot answers.
-   **Tree-of-Thought:** Use for exploring complex problems with multiple possible solutions.
-   **ReAct:** Use for dynamic problems that require information gathering.

Often, the most powerful applications will **chain** these strategies together. For example, you might use a Tree-of-Thought approach where each "thought" is a ReAct loop. By mastering these patterns, you are truly engineering with AI, not just prompting it.

# References and Further Reading

- Unlocking the Power of LLMs: A Guide to Advanced Prompting Strategies (LinkedIn). https://www.linkedin.com/pulse/unlocking-power-llms-guide-advanced-prompting-strategies-srikanth-r-xdqnc
- From Zero to Hero: The Complete Evolution of a Prompt (DEV.to). https://dev.to/jammy_lee_88c9258df43557f/from-zero-to-hero-the-complete-evolution-of-a-prompt-16kd
- My Prompt Stack for Work: 16 Prompts In My AI Toolkit That Make Work a LOT Easier (Substack). https://natesnewsletter.substack.com/p/my-prompt-stack-for-work-16-prompts
- Mastering Veo 3: An Expert Guide to Optimal Prompt Structure and Cinematic Camera Control (Medium). https://medium.com/@miguelivanov/mastering-veo-3-an-expert-guide-to-optimal-prompt-structure-and-cinematic-camera-control-693d01ae9f8b