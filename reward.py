import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [0.5 if match else 0.0 for match in matches]


def accuracy_reward(completions, answers, **kwargs):
    pattern = r".*?<answer>(.*?)</answer>"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match and match.group(1).strip() == answer else 0.0 for match, answer in zip(matches, answers)]


# from https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
def math_accuracy_reward(completions, answers, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, sol in zip(completions, answers):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            # print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def gsm8k_reward(completions, answers, **kwargs):
    return {
        "reward_format": format_reward(completions=completions, answers=answers, **kwargs),
        "reward_accuracy": accuracy_reward(completions=completions, answers=answers, **kwargs),
    }


def huggingface_math_reward(completions, answers, **kwargs):
    return {
        "reward_format": format_reward(completions=completions, answers=answers, **kwargs),
        "reward_accuracy": math_accuracy_reward(completions=completions, answers=answers, **kwargs),
    }


REWARD_FUNCTIONS = {
    "gsm8k_reward": gsm8k_reward,
    "huggingface_math_reward": huggingface_math_reward,
    # "deepscaler_reward": deepscaler_reward
}


if __name__ == "__main__":
    test_completions = [
        "<think>.*?</think>\n<answer>1</answer>dddd"
    ]
    test_answers = [
        "1"
    ]
    print(huggingface_math_reward(test_completions, test_answers))

    contents = [
        r'Since $\begin{pmatrix} a \\ b \end{pmatrix}$ actually lies on $\ell,$ the reflection takes this vector to itself. [asy] unitsize(1.5 cm); pair D = (4,-3), V = (2,1), P = (V + reflect((0,0),D)*(V))/2; draw((4,-3)/2--(-4,3)/2,dashed); draw((-2,0)--(2,0)); draw((0,-2)--(0,2)); draw((0,0)--P,Arrow(6)); label("$\ell$", (4,-3)/2, SE); [/asy] Then \[\begin{pmatrix} \frac{7}{25} & -\frac{24}{25} \\ -\frac{24}{25} & -\frac{7}{25} \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} a \\ b \end{pmatrix}.\]This gives us \[\begin{pmatrix} \frac{7}{25} a - \frac{24}{25} b \\ -\frac{24}{25} a - \frac{7}{25} b \end{pmatrix} = \begin{pmatrix} a \\ b \end{pmatrix}.\]Then $\frac{7}{25} a - \frac{24}{25} b = a$ and $-\frac{24}{25} a - \frac{7}{25} b = b.$ Either equation reduces to $b = -\frac{3}{4} a,$ so the vector we seek is $\boxed{\begin{pmatrix} 4 \\ -3 \end{pmatrix}}.$'
    ]
    print(math_accuracy_reward([r"\boxed{\begin{pmatrix} 4 \\ -3 \end{pmatrix}}"], contents))
