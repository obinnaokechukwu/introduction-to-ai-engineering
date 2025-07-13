# Chapter 2: The Engine Room of AI: Tokens, Embeddings, and Context Windows

In the last chapter, we had our first conversation with an AI. It felt simple, almost magical. But as developers, we know that magic is just well-executed technology. To build powerful, reliable, and cost-effective AI applications, we need to look under the hood. We don't need to become mechanics, but we do need to understand the basic "physics" that govern how these models operate.

Think of an LLM as a high-performance engine. To drive it well, you need to understand three things:

1.  **The Fuel (Tokens):** How the engine consumes fuel and what it costs. For an LLM, the fuel is text, measured in units called **tokens**.
2.  **The Transmission (Embeddings):** How the engine translates your intent into motion. For an LLM, this is the process of turning words into meaningful numbers called **embeddings**.
3.  **The Gas Tank (Context Window):** How far you can go on a single trip before needing to refuel. For an LLM, this is its short-term memory, known as the **context window**.

Mastering these three concepts is the key to moving from simple prompts to sophisticated AI engineering.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Explain what tokens are and how they directly impact your API costs.
-   Use `tiktoken` to count tokens and estimate the cost of an AI call before you make it.
-   Grasp how embeddings allow an AI to understand the *meaning* of words, not just the words themselves.
-   Build a semantic search tool that finds related concepts using embeddings.
-   Understand the limitations of an LLM's "memory" and how to manage a conversation's context window.

## Tokens: The Currency of Language Models

The first and most important thing to understand is that LLMs don't see words or characters. They see the world in **tokens**. A token is a chunk of text, which could be a whole word, part of a word, a number, or a piece of punctuation.

Let's see this in action. The sentence "Hello world!" is broken down by the AI into three distinct tokens: `["Hello", " world", "!"]`.

This might seem like a minor detail, but it is the single most important factor affecting the cost and performance of your application.

### Why Tokens Matter: The Cost of Conversation

AI providers don't charge you per request or per word; they charge you **per token**. This applies to both the tokens you send in your prompt *and* the tokens the AI generates in its response.

A short prompt is cheap. A long one is expensive. Let's make this tangible. We can use a library from OpenAI called `tiktoken` to see exactly how an LLM "sees" our text.

First, install the library:
```bash
pip install tiktoken
```

Now, let's count the tokens in a few different strings.

```python
import tiktoken

# Load the encoder for a specific model. Different models have different tokenizers.
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

text_to_analyze = "This is a simple sentence."

# The .encode() method converts our string into a list of token integers.
tokens = encoder.encode(text_to_analyze)
token_count = len(tokens)

print(f"Text: '{text_to_analyze}'")
print(f"Token count: {token_count}")
print(f"Token IDs: {tokens}")
```

**Output:**

```
Text: 'This is a simple sentence.'
Token count: 6
Token IDs: [2128, 318, 257, 3334, 16934, 13]
```

The text was broken into six tokens. Now, let's build a simple cost calculator to see the financial implications.

```python
# A pricing example for gpt-4o-mini (prices change, check OpenAI's site for current rates)
# Let's assume input costs $0.15 per 1 million tokens.
COST_PER_1000_INPUT_TOKENS = 0.00015

def calculate_prompt_cost(text: str):
    """Estimates the cost of sending a prompt to the OpenAI API."""
    token_count = len(encoder.encode(text))
    cost = (token_count / 1000) * COST_PER_1000_INPUT_TOKENS
    
    print(f"Text: '{text}'")
    print(f"Tokens: {token_count}")
    print(f"Estimated Cost: ${cost:.8f}")
    print("-" * 20)

# Compare a vague prompt with a specific, concise one.
vague_prompt = "Please help me, my IoT device is not working as I expected and I believe there might be an issue with the sensor readings."
concise_prompt = "Help interpret an abnormal IoT sensor reading."

calculate_prompt_cost(vague_prompt)
calculate_prompt_cost(concise_prompt)
```

You'll see that the vague, wordy prompt can be five to six times more expensive than the concise one, even though they are asking for the same thing.

> **Key Takeaway:** As an AI developer, your first optimization challenge is "token engineering." Be as concise as possible in your prompts without losing necessary context. Every token saved is money in your pocket.

## Embeddings: How AI Understands Meaning

Now that we know how AI *counts* text, let's explore how it *understands* it. The magic behind this is a concept called **embeddings**.

An embedding is a way of turning a token or a piece of text into a list of numbers—a vector—that captures its semantic meaning.

Think of it like a map. On a world map, we can represent any city with two numbers: latitude and longitude. Cities that are geographically close, like London and Paris, will have similar coordinates.

Embeddings do the same thing, but for meaning. They place words in a high-dimensional "meaning space."

-   The vector for "cat" will be very close to the vector for "kitten."
-   The vector for "happy" will be very close to the vector for "joyful."
-   The vector for "car" will be far away from the vector for "planet."

This allows the AI to understand relationships between words. Even more powerfully, the *relationship* between vectors is also meaningful. The vector from "man" to "woman" is very similar to the vector from "king" to "queen."

### Creating Your First Embedding

Let's get the embedding vector for a piece of text. The `text-embedding-3-small` model is a great, cost-effective choice for this.

```python
import openai

client = openai.OpenAI()

def get_embedding(text: str) -> list[float]:
    """Generates an embedding vector for a given piece of text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Get the embedding for a simple word
cat_embedding = get_embedding("cat")

print(f"The word 'cat' is represented by a list of {len(cat_embedding)} numbers.")
print(f"Here are the first 5 numbers: {cat_embedding[:5]}")
```

The output is a list of 1536 numbers. We don't care about the specific values; we only care about comparing them to other vectors.

### A Practical Use Case: Semantic Search

The most powerful application of embeddings is **semantic search**—finding things based on meaning, not just keywords. Let's build a tool that can find the most relevant troubleshooting step for a user's problem.

```python
import numpy as np

def find_most_similar(query: str, options: list[str]) -> str:
    """Finds the option most semantically similar to the query."""
    
    # 1. Get embeddings for the query and all options in a single API call
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query] + options
    )
    
    # 2. Separate the query embedding from the option embeddings
    query_embedding = response.data[0].embedding
    option_embeddings = [item.embedding for item in response.data[1:]]
    
    # 3. Calculate the similarity between the query and each option
    # We use cosine similarity, which is a dot product of normalized vectors.
    similarities = [np.dot(query_embedding, opt_emb) for opt_emb in option_embeddings]
    
    # 4. Find the index of the highest similarity score
    best_option_index = np.argmax(similarities)
    
    return options[best_option_index]

# Our knowledge base of potential solutions
possible_solutions = [
    "Check for direct sunlight hitting the sensor.",
    "Restart the device's WiFi router.",
    "Recalibrate the sensor against a known temperature source.",
    "Replace the device's battery.",
]

# A user's problem, described in their own words
user_problem = "The thermostat reading is way too high, it says 95 degrees in here!"

# Find the best solution
best_solution = find_most_similar(user_problem, possible_solutions)

print(f"User's Problem: '{user_problem}'")
print(f"Most Relevant Solution: '{best_solution}'")
```

Notice that the user's problem doesn't contain the keywords "sunlight" or "direct." But because the *meaning* of "thermostat reading is way too high" is semantically close to the concept of an external heat source like sunlight, the embedding-based search finds the correct solution. This is far more powerful than a simple keyword search.

## Context Windows: The AI's Limited Memory

The final core concept is the **context window**. This is arguably the most important limitation to understand when building applications.

An LLM's context window is its **short-term memory**. It's the maximum number of tokens the model can consider at one time when generating a response. Everything—the system prompt, the user prompts, and the assistant's own past responses—must fit within this window.

If a conversation exceeds the context window, the model starts to "forget" the earliest messages.

### Context Window Sizes of Popular Models

The size of the context window is a key feature that differentiates models. Larger windows mean the AI can "remember" longer conversations or process larger documents.

| Model           | Context Window (Tokens) | Approximate Pages of Text |
| --------------- | ----------------------- | ------------------------- |
| `gpt-4o-mini`     | 128,000                 | ~250 pages                |
| `gpt-4-turbo`     | 128,000                 | ~250 pages                |
| `claude-3-5-sonnet` | 200,000                 | ~400 pages                |
| `gemini-1.5-pro`  | 1,000,000               | ~2000 pages               |

Even with large windows, managing context is crucial for performance and cost. Sending the entire history of a long conversation in every API call is slow and expensive.

### A Simple Memory Management Strategy

A common strategy for managing long conversations is to implement a "sliding window" of memory. We always keep the system prompt, but we start removing the oldest user/assistant messages once we approach the token limit.

```python
def manage_conversation_history(messages: list[dict], max_tokens: int = 120000) -> list[dict]:
    """Ensures the conversation history fits within the token limit."""
    
    # Always keep the system message if it exists
    if messages and messages[0]['role'] == 'system':
        system_message = messages.pop(0)
    else:
        system_message = None

    # Calculate token usage and remove old messages until it fits
    current_tokens = sum(len(encoder.encode(msg['content'])) for msg in messages)
    while current_tokens > max_tokens:
        # Remove the oldest message
        removed_message = messages.pop(0)
        current_tokens -= len(encoder.encode(removed_message['content']))

    # Add the system message back to the beginning
    if system_message:
        messages.insert(0, system_message)
        
    return messages
```

This function provides a basic safety net to prevent your API calls from failing due to an oversized context.

## What You've Learned

You now understand the three fundamental concepts that govern how you will build with LLMs.

1.  **Tokens are the currency.** Every interaction has a cost. Be concise to save money.
2.  **Embeddings capture meaning.** This allows for powerful semantic search and relevance ranking, far beyond simple keyword matching.
3.  **Context windows are the memory.** All models have a finite memory, and you must manage it to handle long conversations and large documents.

These three ideas—cost, meaning, and memory—are the levers you will constantly be pulling and balancing as an AI developer. In the next chapter, we will get our hands dirty and set up a professional development environment so you can start building real applications on this foundation.

# References and Further Reading

- AI Tokens Explained: Understanding Context Windows and Processing. https://medium.com/@jimcanary/ai-tokens-explained-understanding-context-windows-and-processing-a99ca2dd9142
- What Are AI Tokens and Context Windows (And Why Should You Care)? https://simple.ai/p/tokens-and-context-windows
- From Tokens to Context Windows: Simplifying AI Jargon. Technology Policy Institute. https://techpolicyinstitute.org/publications/artificial-intelligence/from-tokens-to-context-windows-simplifying-ai-jargon/
- AI Tokens Explained - What They Are and Why They Matter. https://zenvanriel.nl/ai-engineer-blog/ai-tokens-explained-what-they-are-and-why-they-matter/