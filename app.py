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

    messages = [

        {
            "role": "system",

            "content": """
You are an AI task management agent.

Your job is to understand the user's request and use
available tools when necessary.

Available actions:

1. Calculator
2. Add task
3. List tasks
4. Remove task

IMPORTANT RULES:

- Use tools whenever an action is required.
- Never claim an action was completed unless the tool
  actually executed it.
- Continue using tools when multiple actions are required.
- Only provide the final response after completing the
  required actions.
"""
        },

        {
            "role": "user",

            "content": user_request
        }

    ]

    max_steps = 5

    for step in range(max_steps):

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            tools=tools,

            tool_choice="auto",

            max_tokens=400
        )

        assistant_message = response.choices[0].message

        # No more tools required

        if not getattr(
            assistant_message,
            "tool_calls",
            None
        ):

            return (
                assistant_message.content,
                []
            )

        messages.append(
            assistant_message
        )

        tool_activity = []

        for tool_call in assistant_message.tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = json.loads(
                tool_call.function.arguments
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

            messages.append(
                {
                    "role": "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        str(result)
                }
            )

    return (
        "The agent reached the maximum number of tool steps.",
        []
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
