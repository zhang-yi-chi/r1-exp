from torch.utils.data import Dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def process_gsm8k_answer(answer):
    if "####" in answer:
        return answer.split("####")[1].strip()
    else:
        return answer


def preprocess_data(data, input_template=None, input_key="input", answer_key="answer", apply_chat_template=None) -> str:
    # TODO: config process_answer
    answer = process_gsm8k_answer(data[answer_key])
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chat}
            ]
        processed_data = apply_chat_template(chat, tokenize=False, add_generation_prompt=True), answer
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        processed_data = prompt, answer
    return processed_data


class R1Dataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        answer_key = getattr(self.strategy.args, "answer_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.data_list = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            processed_data = preprocess_data(data, input_template, input_key, answer_key, apply_chat_template)
            self.data_list.append(processed_data)

    def __len__(self):
        length = len(self.data_list)
        return length

    def __getitem__(self, idx):
        return self.data_list[idx]
