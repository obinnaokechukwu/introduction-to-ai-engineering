# Chapter 3: Setting Up Your Development Environment

Every great journey begins with a first step, and in software development, that first step is setting up a clean, secure, and reproducible environment. This chapter is your guide to building a solid foundation for all the AI projects to come. We'll move methodically, ensuring that from day one, you're following best practices for managing your projects, code, and—most importantly—your secret API keys.

### Learning Objectives

By the end of this chapter, you will be able to:

-   Configure a Python environment specifically for AI development.
-   Install the essential libraries required for interacting with AI models.
-   Securely manage and use your API keys without ever exposing them in your code.
-   Create a well-structured project folder that can be used as a template for future work.
-   Build and run your first complete, command-line AI chatbot.

## What You'll Need

This guide is designed to be accessible. You don't need to be a Python expert, but some basic familiarity with the command line will be helpful. Here’s all you need to get started:

1.  A computer (Windows, macOS, or Linux).
2.  An internet connection.
3.  The ability to open and use a terminal or command prompt.

## The Foundation: Installing and Configuring Python

Modern AI development is powered by Python. Its clean syntax and vast ecosystem of libraries make it the language of choice for machine learning and AI.

### Step 1: Do You Have Python?

First, let's check if Python is already installed on your system. Open your terminal (Terminal on macOS/Linux, Command Prompt or PowerShell on Windows) and type the following command:

```bash
python3 --version
```

If you see a version number like `Python 3.9.x` or higher, you're in great shape. If you get an error or the version is older than 3.9, you'll need to install or update it.

### Step 2: Installing Python (If Necessary)

-   **Windows:** Go to the official [python.org](https://www.python.org/downloads/) website, download the latest stable version (e.g., Python 3.11), and run the installer. **Crucially, on the first screen of the installer, check the box that says "Add Python to PATH."** This will make it accessible from your command prompt.
-   **macOS:** The easiest way is with Homebrew, a package manager for macOS.
    ```bash
    # Install Homebrew (if you don't have it)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Install Python
    brew install python
    ```
-   **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-venv
    ```

### Step 3: Creating Your First AI Project Directory

Organization is key. Let's create a dedicated folder for all your AI work.

```bash
# In your terminal, navigate to where you keep your projects
mkdir my_ai_projects
cd my_ai_projects

# Now, create the directory for our first project
mkdir simple-chatbot
cd simple-chatbot
```

### Step 4: The Power of Virtual Environments

A **virtual environment** is one of the most important best practices in Python development. It creates an isolated "bubble" for each project, so the libraries you install for one project don't interfere with others.

Let's create one for our chatbot.

```bash
# This command creates a new virtual environment folder named 'venv'
python3 -m venv venv
```

Now, you need to **activate** it. The command differs slightly by operating system:

-   **macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```
-   **Windows (Command Prompt):**
    ```bash
    venv\Scripts\activate
    ```

Once activated, you'll see `(venv)` at the beginning of your terminal prompt. This tells you that you are now working inside your isolated project environment.

## Installing the Essential AI Libraries

With our environment ready, we can install the necessary packages. We'll start with just two:

1.  `openai`: The official Python library for interacting with OpenAI's models (like GPT-4).
2.  `python-dotenv`: A utility for managing secret keys securely.

```bash
# Make sure your virtual environment is active!
pip install openai python-dotenv
```

That's it for now. You have the core tools needed to build your first application.

## API Key Security: Your Most Important Lesson

Your API key is like a password to your AI provider account. It's linked to your billing information, and anyone who has it can make API calls on your behalf, costing you money. **You must never, ever, put your API key directly into your code or commit it to a public repository like GitHub.**

Here is the professional, secure way to handle it.

### Step 1: Get Your API Key

1.  Navigate to [platform.openai.com](https://platform.openai.com).
2.  Sign up or log in.
3.  Click on your profile icon in the top-right and go to the "API keys" section.
4.  Click "Create new secret key," give it a name (e.g., "MyFirstAIApp"), and create the key.
5.  **Copy the key immediately and save it somewhere safe.** You will not be able to see the full key again after you close the window.

### Step 2: Store Your Key in a `.env` File

Inside your `simple-chatbot` project directory, create a new file named `.env`. This file will hold your secret keys.

```
# In your .env file
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Replace `sk-xxx...` with the actual key you just copied.

### Step 3: Tell Git to Ignore Your Secrets

It's critically important that your `.env` file is never uploaded to version control. We do this by creating a `.gitignore` file.

```bash
# In your terminal, inside the project directory
# This command creates a .gitignore file and adds '.env' to it.
echo ".env" >> .gitignore
# It's also good practice to ignore your virtual environment folder.
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
```

Now, Git will completely ignore these files, keeping your secrets safe.

### Step 4: Use the Key Securely in Your Code

The `python-dotenv` library makes it easy to load the variables from your `.env` file into your application's environment.

Create a file named `main.py` and add the following code:

```python
# main.py
import openai
import os
from dotenv import load_dotenv

# This line loads the variables from your .env file
load_dotenv()

# os.getenv() securely reads the environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not found. Please check your .env file.")
else:
    # Create the OpenAI client with your key
    client = openai.OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized successfully!")
    
    # Let's test it with a real API call
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Confirm you are operational."}]
        )
        print("🎉 AI is responding! Your setup is complete.")
        print(f"AI: {response.choices[0].message.content}")
    except Exception as e:
        print(f"An error occurred: {e}")
```

Run your script from the terminal:

```bash
python main.py
```

If everything is configured correctly, you will see a confirmation message from the AI. You now have a secure, working development setup!

## Building Your First Complete Project: An Interactive Chatbot

Let's use our new setup to build a complete, albeit simple, application. We will create a command-line chatbot that remembers the conversation history.

### Final Project Structure

Your `simple-chatbot` directory should now look like this:

```
simple-chatbot/
├── venv/                 # Your virtual environment
├── .env                  # Your secret API key
├── .gitignore            # Files for Git to ignore
├── main.py               # Our main application code
└── requirements.txt      # A file listing our project's libraries
```

### Create `requirements.txt`

It's a best practice to list your project's dependencies in a `requirements.txt` file. This allows you or anyone else to easily install the correct libraries for the project.

```bash
# Run this command to generate the file automatically
pip freeze > requirements.txt
```

### The `main.py` Chatbot Code

Let's rewrite `main.py` to be a reusable class and interactive loop.

```python
# main.py
import openai
import os
from dotenv import load_dotenv

class Chatbot:
    def __init__(self):
        # Load environment variables and initialize the client
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize conversation history with a system message
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def have_conversation(self, user_input: str) -> str:
        """Sends user input to the AI and gets a response."""
        
        # Add the user's message to the history
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # Make the API call with the entire conversation history
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversation_history
            )
            
            # Extract the AI's response message
            ai_message = response.choices[0].message.content
            
            # Add the AI's response to the history for future context
            self.conversation_history.append({"role": "assistant", "content": ai_message})
            
            return ai_message
        
        except Exception as e:
            return f"An error occurred: {e}"

    def start(self):
        """Starts the interactive chat loop."""
        print("🤖 AI Chatbot is ready. Type 'exit' or 'quit' to end the conversation.")
        while True:
            try:
                user_message = input("You: ")
                if user_message.lower() in ["exit", "quit"]:
                    print("👋 Goodbye!")
                    break
                
                ai_response = self.have_conversation(user_message)
                print(f"AI: {ai_response}")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

if __name__ == "__main__":
    try:
        bot = Chatbot()
        bot.start()
    except ValueError as e:
        print(f"Configuration Error: {e}")

```

Now, run your complete chatbot:

```bash
python main.py
```

You can now have a continuous conversation with your AI assistant. Ask it a question, and then ask a follow-up. It will remember the context because we are sending the entire `conversation_history` with each new request.

## What You've Accomplished

Congratulations! You have successfully set up a professional-grade development environment for building AI applications. You've learned to:

-   Create isolated Python projects using virtual environments.
-   Install the necessary AI libraries.
-   Handle your secret API keys securely using `.env` files and environment variables—the single most important safety practice.
-   Structure your project with a `requirements.txt` file for reproducibility.
-   Build and run a complete, conversational AI application.

This solid foundation is not just for learning; it's the same setup professionals use to build real-world products. You are now ready to move beyond the basics and start exploring the more advanced capabilities of AI models.

# References and Further Reading

- Setting up an AI/ML Engineering Dev Environment (Chris Johnson, Indeed). https://www.linkedin.com/pulse/setting-up-aiml-engineering-dev-environment-chris-johnson-cm9hc
- Setting Up a Python Environment for AI Development: A Comprehensive Guide. https://medium.com/@flexianadevgroup/setting-up-a-python-environment-for-ai-development-a-comprehensive-guide-022602e337f4
- Getting Started with AI Development: Setting Up Your Python Environment. https://dev.to/sahilmadhyan/getting-started-with-ai-development-setting-up-your-python-environment-275o
- The Ultimate Deep Learning Project Structure: A Software Engineer’s Guide into the Land of AI. https://ai.plainenglish.io/the-ultimate-deep-learning-project-structure-a-software-engineers-guide-into-the-land-of-ai-c383f234fd2f