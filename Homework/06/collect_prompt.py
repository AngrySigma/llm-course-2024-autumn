def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    question = sample["question"]
    subject = sample["subject"]
    choices = sample["choices"]

    return (f"The following are multiple choice questions (with answers) about {subject}.\n"
        f"{question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:")


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    question = sample["question"]
    subject = sample["subject"]
    choices = sample["choices"]

    prompt = ""

    for example in examples:
        prompt += (f"The following are multiple choice questions (with answers) about {subject}.\n"
                    f"{example['question']}\n"
                   f"A. {example['choices'][0]}\n"
                   f"B. {example['choices'][1]}\n"
                   f"C. {example['choices'][2]}\n"
                   f"D. {example['choices'][3]}\n"
                   f"Answer: {chr(65 + example['answer'])}"
                   )
        prompt += f". {example['choices'][example['answer']]}\n\n" if add_full_example else "\n\n"


    prompt += (f"The following are multiple choice questions (with answers) about {subject}.\n"
                f"{question}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                "Answer:")

    return prompt
