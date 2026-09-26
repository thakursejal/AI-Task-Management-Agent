import streamlit as st
from huggingface_hub import InferenceClient
import json

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Task Management Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Task Management Agent")
st.caption("An AI agent that understands requests, selects tools, performs actions, and responds.")

# =========================================================
# HUGGING FACE CLIENT
# =========================================================

HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL = "openai/gpt-oss-120b:fastest"

# =========================================================
# TASK STATE
# =========================================================

if "tasks" not in st.session_state:
    st.session_state.tasks = []


# =========================================================
# TOOLS
# =========================================================

def calculator(expression):

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return f"Calculation result: {result}"

    except Exception:
        return "Unable to calculate the expression."


def add_task(task):

    st.session_state.tasks.append(task)

    return f"Task added successfully: {task}"


def list_tasks():

    if not st.session_state.tasks:
        return "No tasks found."

    result = "Current tasks:\n"

    for i, task in enumerate(
        st.session_state.tasks,
        start=1
    ):
        result += f"{i}. {task}\n"

    return result


def remove_task(task_number):

    try:

        task_number = int(task_number)

        if 1 <= task_number <= len(st.session_state.tasks):

            removed = st.session_state.tasks.pop(
                task_number - 1
            )

            return f"Task removed: {removed}"

        return "Invalid task number."

    except Exception:

        return "Invalid task number."


# =========================================================
# TOOL DEFINITIONS
# =========================================================

tools = [

    {
        "type": "function",
        "function": {

            "name": "calculator",

            "description":
                "Calculate a mathematical expression.",

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {

                        "type": "string",

                        "description":
                            "Mathematical expression to calculate."

                    }

                },

                "required": ["expression"]
            }
        }
    },

    {
        "type": "function",
        "function": {

            "name": "add_task",

            "description":
                "Add a new task to the task list.",

            "parameters": {

                "type": "object",

                "properties": {

                    "task": {

                        "type": "string",

                        "description":
                            "Task that should be added."

                    }

                },

                "required": ["task"]
            }
        }
    },

    {
        "type": "function",
        "function": {

            "name": "list_tasks",

            "description":
                "Show all current tasks.",

            "parameters": {

                "type": "object",

                "properties": {}

            }
        }
    },

    {
        "type": "function",
        "function": {

            "name": "remove_task",

            "description":
                "Remove a task using its task number.",

            "parameters": {

                "type": "object",

                "properties": {

                    "task_number": {

                        "type": "integer",

                        "description":
                            "Number of the task to remove."

                    }

                },

                "required": ["task_number"]
            }
        }
    }
]


# =========================================================
# TOOL EXECUTION
# =========================================================

def execute_tool(
    tool_name,
    arguments
):

    if tool_name == "calculator":

        return calculator(
            arguments["expression"]
        )

    elif tool_name == "add_task":

        return add_task(
            arguments["task"]
        )

    elif tool_name == "list_tasks":

        return list_tasks()

    elif tool_name == "remove_task":

        return remove_task(
            arguments["task_number"]
        )

    return "Unknown tool."


# =========================================================
# AI AGENT
# =========================================================

def run_agent(user_request):

    # Ask the LLM to decide which actions are needed.
    decision_prompt = f"""
You are an AI task management agent.

Analyze the user's request and decide which actions are required.

Available tools:

1. calculator
   Arguments:
   {{"expression": "mathematical expression"}}

2. add_task
   Arguments:
   {{"task": "task text"}}

3. list_tasks
   Arguments:
   {{}}

4. remove_task
   Arguments:
   {{"task_number": 1}}

Return ONLY valid JSON.

Use this format:

{{
  "actions": [
    {{
      "tool": "tool_name",
      "arguments": {{}}
    }}
  ]
}}

If no tool is required, return:

{{
  "actions": []
}}

User request:
{user_request}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise AI agent that selects tools."
            },
            {
                "role": "user",
                "content": decision_prompt
            }
        ],
        max_tokens=300
    )

    decision_text = response.choices[0].message.content.strip()

    # Remove Markdown JSON fences if the model adds them
    decision_text = decision_text.replace(
        "```json", ""
    ).replace(
        "```", ""
    ).strip()

    try:

        decision = json.loads(decision_text)

    except Exception:

        return (
            "I could not understand the requested action.",
            []
        )

    actions = decision.get(
        "actions",
        []
    )

    tool_activity = []
    tool_results = []

    # Execute selected tools
    for action in actions:

        tool_name = action.get(
            "tool"
        )

        arguments = action.get(
            "arguments",
            {}
        )

        result = execute_tool(
            tool_name,
            arguments
        )

        tool_activity.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            }
        )

        tool_results.append(
            f"Tool: {tool_name}\n"
            f"Arguments: {arguments}\n"
            f"Result: {result}"
        )

    # If no tool was required
    if not tool_results:

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": user_request
                }
            ],
            max_tokens=300
        )

        return (
            final_response.choices[0].message.content,
            tool_activity
        )

    # Give tool results back to the LLM
    result_context = "\n\n".join(
        tool_results
    )

    final_prompt = f"""
You are an AI task management assistant.

The user requested:

{user_request}

The following actions were actually executed:

{result_context}

Write a concise final response to the user.

IMPORTANT:
Only claim that an action was completed if the tool
result confirms that it was completed.
"""

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        max_tokens=300
    )

    return (
        final_response.choices[0].message.content,
        tool_activity
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🛠️ Available Tools")

    st.write("🧮 Calculator")
    st.write("➕ Add Task")
    st.write("📋 List Tasks")
    st.write("🗑️ Remove Task")

    st.markdown("---")

    st.header("📋 Current Tasks")

    if st.session_state.tasks:

        for i, task in enumerate(
            st.session_state.tasks,
            start=1
        ):

            st.write(
                f"{i}. {task}"
            )

    else:

        st.write("No tasks yet.")


# =========================================================
# CHAT INTERFACE
# =========================================================

st.subheader("💬 Talk to Your AI Agent")

user_request = st.chat_input(
    "Example: Add a task called Complete Module 5"
)

if user_request:

    st.chat_message(
        "user"
    ).write(
        user_request
    )

    with st.spinner(
        "🤖 Agent is thinking and using tools..."
    ):

        answer, activity = run_agent(
            user_request
        )

    st.chat_message(
        "assistant"
    ).write(
        answer
    )

    # Show tool activity

    st.subheader("🔧 Agent Activity")

    st.info(
        "The agent automatically selected and executed "
        "the required tools."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built by Thakur Sejal | Codomax Digital Solutions Internship | Module 5"
)
