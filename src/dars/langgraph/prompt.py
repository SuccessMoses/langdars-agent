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
"""

###########################################
#   INSTANCE
###########################################

instance_template = """
Here's the issue you need to address, as described in the PR:

<pr_description>
{issue}
</pr_description>
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
