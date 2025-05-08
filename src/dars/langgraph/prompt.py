edit_expansion_prompt_template = """
You will be given information about a previous action and its trajectory. Your goal is to suggest a refined or alternative action that better resolves the issue at hand.
Here is the information about the previous modification:

Previous action:
<previous_action>
{action}
</previous_action>

Trajectory after the action:
<previous_trajectory>
{prev_traj}
</previous_trajectory>

Instructions:
1. Analyze the previous action and its trajectory.
2. Suggest a replacement action that improves upon the previous one.
3. Focus on refining the current edit, modifying different sections, or making small insertions as needed.
4. Keep your suggestion concise and directly related to the file modification.

Before providing your final suggestion, wrap your analysis process in <analysis> tags. In this analysis:
1. Summarize the previous action and its trajectory
2. Identify the key issues or shortcomings in the previous action
3. List potential improvements or alternative approaches
4. Consider how these changes might affect the trajectory

You need to format your output using three fields; analysis, discussion and command.
"""

insert_expansion_prompt_template = """
You will be given information about a previous action and its trajectory. Your goal is to suggest a single, concise improvement that replaces the previous action.
Here's the information about the previous modification:

Previous action:
<action>
{action}
</action>

Trajectory after the action:
<prev_traj>
{prev_traj}
</prev_traj>

Your task is to analyze this information and suggest one improvement. This improvement should replace the previous action, not be a next step. Focus on one of these approaches:
1. A different insertion with varied content
2. An insertion in a new location
3. Editing existing content for a more effective resolution

Before providing your final suggestion, wrap your analysis process in <analysis> tags. In this analysis:
1. Summarize the previous action and its trajectory
2. Identify the key issues or shortcomings in the previous action
3. List potential improvements or alternative approaches
4. Consider how these changes might affect the trajectory

You need to format your output using three fields; analysis, discussion and command.
"""

append_expansion_prompt_template = """
Your goal is to suggest alternative content for appending to a file, based on a previous action and its outcome.
Here's the information about the previous operation:

<previous_action>
{action}
</previous_action>

<previous_trajectory>
{prev_traj}
</previous_trajectory>

Your task is to suggest a replacement for the previous append action, not to provide the next action in the sequence. The reproduction script you've written may lack completeness on its own. Would you like to review it and write a more comprehensive version of the script, incorporating the context of the previous trajectory?
1. Analyze the previous action:
  - What specific content was appended?
  - What was the likely purpose of this content?

2. Brainstorm at least three alternative content ideas:
  - Describe each alternative and how it differs from the original.
  - Number each alternative for easy reference.

3. Evaluate each alternative:
  - How does it potentially improve exploration?
  - What new insights might it provide?

4. Select the best alternative:
  - Which option do you think is most promising?
  - Justify your choice in 1-2 sentences.

Before providing your final suggestion, wrap your analysis process in <analysis> tags. In this analysis:
1. Summarize the previous action and its trajectory
2. Identify the key issues or shortcomings in the previous action
3. List potential improvements or alternative approaches
4. Consider how these changes might affect the trajectory

You need to format your output using three fields; analysis, discussion and command.
"""

submit_expansion_prompt_template = """
You are about to submit the changes. Have you double-checked that your changes don't affect other test cases or have any unintended consequences or completely fix the issue? Please review once more before submitting.
"""

create_expansion_prompt_template = """
Before trying to reproduce the bug, let's first try to localize the issue, we can test the issue after the fix.
"""

critic_expansion_prompt_template = """
You are an AI system tasked with selecting the best alternative action to replace a previously executed action in a process or workflow. Your goal is to evaluate the given alternatives and choose the most effective replacement.
Here is the previously executed action:
<previous_action>

{previous_action}
</previous_action>

Here is the list of alternative actions to consider:
<alternative_actions>
{actions}
</alternative_actions>

Instructions:
1. Evaluate each action in the list of alternative actions based on the following criteria:
  a. It must be different from the previous action.
  b. It should replace the previous action, not be implemented after it.
  c. It should be more effective than the previous action.

2. Analyze each action inside <action_analysis> tags, following this structure:
  - List each action with a number.
  - For each action, explicitly state whether it meets each of the three criteria.
  - Provide a brief explanation for why the action does or doesn't meet each criterion.
  - If the action meets all criteria, give it a numerical effectiveness score (1-10).

3. After evaluating all actions, select the best one that meets all the criteria and is the most effective replacement for the previous action.

4. Provide the index of the best action using <best_action_index> tags starting from 0.

Example output format:
<action_analysis>
[All actions analysis one by one]
</action_analysis>

<best_action_index>[Your selected best action index]</best_action_index>
"""

############################################
#    SYSTEM
############################################
system_template = """
SETTING: You are an autonomous programmer working in a command-line interface with a special file editor. Your task is to solve issues within a repository by analyzing, editing, and testing code.

Available Commands:
<command_docs>
{command_docs}
search_repo:
  docstring: searches in the current repository with a specific function or class, and returns the def and ref relations for the search term.
  signature: search_repo <search_term>
  arguments:
    - search_term (string) [required]: function or class to look for in the repository.
</command_docs>

General Guidelines:
1. One command at a time: Always execute a single command and wait for feedback before proceeding.
2. Proper indentation: When editing files, ensure correct indentation for each line.
3. File awareness: Pay attention to the currently open file and working directory.
4. Search functionality: Use search_repo command to gather information when needed.
5. For interactive sessions: Start it using `execute_server` command.

You need to format your output using two fields; discussion and command.
Your output should always include _one_ discussion and _one_ command field EXACTLY as in the following example:
DISCUSSION
First I'll start by using ls to see what files are in the current directory. Then maybe we can look at some relevant files to see what they look like.
```
ls -a
```

(Open file: {open_file})
(Current directory: {working_dir})
bash-$
"""

instance_template = """
Here's the issue you need to address, as described in the PR:

<pr_description>
{issue}
</pr_description>

You're in the repository's root directory. Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
Start by creating a minimal script to replicate and verify the bug described in the issue. Ensure the bug is reproducible before making any changes. After implementing a fix, use the same script to confirm the issue is resolved. Include debugging messages, like print("Script completed successfully."), to indicate successful execution. The script should be focused on verification and ensuring no new errors are introduced.
Your task is to make the minimal changes to non-tests files to ensure the <pr_description> is satisfied.
If a command fails, do not repeat it. It will not work the second time unless you modify it. Always adapt or use a different command.
Note: Please give only single tool call in a single step.

Follow these steps to resolve the issue:

1. Explore the repository structure to familiarize yourself with its layout.
2. Create a script to reproduce the error and execute it using the BashTool.
3. Edit the source code to resolve the issue, making minimal changes.
4. Rerun your reproduce script to confirm the error is fixed.
5. Consider edge cases and ensure your fix handles them.

Important Instructions for Command Usage:

1. File Navigation:
  - Always be aware of the currently open file and the current working directory.
  - The currently open file might be in a different directory than the working directory.
  - Some commands, like 'create', may change the current open file.
  - For efficient navigation to specific lines (e.g., line 583), use 'goto' instead of multiple scroll_down commands.

2. Code Editing Commands (edit, append, insert):
  - If the assistant would like to add the line '        print(x)', it must fully write the line out, with all leading spaces before the code!
  - Prefix `content` with `$` to ensure the string is treated as a literal, avoiding the need for escape characters.
  - Use $'...' Notation: Always use $'...' for strings in edit, append, and insert commands to correctly interpret escape sequences like \n. Avoid $"...", as it treats escape sequences literally.
  - Example for Clarity: For instance, use append $'line1\nline2\n...' instead of append $"line1\nline2\n..." to ensure \n is interpreted as a newline.
  - To add characters like `\n` or `\t` as literal strings (not as newlines or tabs) within code, or to edit existing code with these characters, use double backslashes (e.g., "\\n", "\\t").
  - Escape single or double quotes within code as \' or \".
  - Escape characters are generally unnecessary (except for the specific cases noted above) because using `$` before content ensures correct interpretation by default. Simply provide code strings as they appear, without additional escapes.
  - Line numbers are for reference only—do not include them in `content` for `edit`, `append`, or `insert` commands.
  - Avoid adding comments unless absolutely necessary to explain non-obvious behavior.

3. Edit Command:
  - The `to_replace` argument must exactly match the existing source code, character for character, including all comments, docstrings, and indentation.
  - Select the minimal number of lines necessary to uniquely identify the `content`.
  - Prefix `to_replace` and `new_content` with `$` to ensure the string is treated as a literal.
  - Ensure `new_content` includes correct indentation.
  - To remove lines, set `new_content` to an empty string.
  - Note that `to_replace` and `new_content` must be different.
  - Ensure `to_replace` and `new_content` contain the full line(s), including indentation and comments, for accurate editing.
    * For example, if replacing `    a = f(x) + g(y) + t`    a = k(x) + g(y) + t`, use `edit $'    a = f(x) + g(y) + t' $'    a = k(x) + g(y) + t'` rather than partial matches like `edit $'a = f(x)' $'b = k(x)'`.

4. Insert Command:
  - Specify the exact line number for insertion.
  - This command will not modify content before or after the specified line.

5. Append Command:
  - Use `append` to add content to the end of a file without modifying any existing lines.
  - This is ideal after a `create_file` command.

6. Search Command:
  - `search_repo` searches the current repository for specified functions or classes.
  - It provides definition (def) and reference (ref) relationships for the search term.
  - `search_term` is the function or class name to search for.

7. Execute_server Command:
  - Use the `execute_server` command to run a server or process in the background.
  - Usage: `execute_server 'Your_Command'`. Make sure to use quotes around the command.
  - `execute_server get_logs`: Retrieves the last 100 lines of the server / process logs.
  - `execute_server stop`: Stops the background Bash server process.
"""


############################################
#    ENV
############################################

next_step_template = """
{observation}
(Open file: {open_file})
(Current directory: {working_dir})
bash-$
"""

next_step_no_output_template = """
Your command ran successfully and did not produce any output.
(Open file: {open_file})
(Current directory: {working_dir})
bash-$
"""

next_step_codegraph_template = """
Your command ran successfully and produced the following related functions/classes for {search_term}:
For each item, `fname` denotes the source file, `line` denotes the line number, `kind` means whether it is definition or reference, and `info` contains the specific content.
{codegraph_context}
(Open file: {open_file})
(Current directory: {working_dir})
bash-$
"""
