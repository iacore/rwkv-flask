"""
example only, not executable

from Alpaca Instruct dataset

you can also prompt the model in other ways
"""

instruction = "TK"
input = "TK"

prompt_with_input =\
f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Input:
{input}

# Response:
"""

prompt_without_input =\
f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""
