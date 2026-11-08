"""A minimal agent: Claude + two tools + a loop you can read in one sitting."""

from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from your .env file
client = anthropic.Anthropic()

MODEL = "claude-sonnet-5-5"  # check the models page in Anthropic's docs for current names
MAX_STEPS = 10  # safety limit so a confused agent can't loop (and bill you) forever
NOTES_DIR = Path("notes")

# 1. Describe the tools to Claude. Claude never runs these itself; it only
#    *asks* for them by name, and your code does the actual work.
TOOLS = [
    {
        "name": "read_file",
        "description": "Read a text file from the current project folder and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path, e.g. 'hello.py'"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "save_note",
        "description": "Save a note as a markdown file in the notes/ folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "e.g. 'summary.md'"},
                "content": {"type": "string", "description": "The note text"},
            },
            "required": ["filename", "content"],
        },
    },
]


# 2. The real Python functions behind those tools.
def read_file(path: str) -> str:
    target = Path(path).resolve()
    if Path.cwd().resolve() not in target.parents:  # keep the agent inside this folder
        return "Error: can only read files inside the project folder."
    if target.name == ".env":
        return "Error: that file is off limits."
    if not target.is_file():
        return f"Error: {path} not found."
    return target.read_text()


def save_note(filename: str, content: str) -> str:
    NOTES_DIR.mkdir(exist_ok=True)
    (NOTES_DIR / Path(filename).name).write_text(content)  # .name blocks path tricks
    return f"Saved notes/{Path(filename).name}"


def run_tool(name: str, args: dict) -> str:
    try:
        if name == "read_file":
            return read_file(**args)
        if name == "save_note":
            return save_note(**args)
        return f"Error: unknown tool {name}"
    except Exception as e:  # return errors to Claude so it can adapt
        return f"Error: {e}"


# 3. The agent loop: ask Claude, run any tools it requests, feed results back, repeat.
def run_agent(task: str) -> None:
    messages = [{"role": "user", "content": task}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # No tool requested means Claude is done.
        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text":
                    print(f"\nFinal answer:\n{block.text}")
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[step {step}] Claude called {block.name}({block.input})")
                output = run_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": tool_results})

    print("Stopped: hit MAX_STEPS without finishing.")


if __name__ == "__main__":
    run_agent(
        "Read hello.py, explain what it does in two sentences, "
        "and save that explanation as a note called summary.md."
    )
