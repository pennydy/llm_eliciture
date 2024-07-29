import numpy as np
import pandas as pd
import logging
import random
from openai import OpenAI
from tqdm import tqdm
import argparse
import os

logger = logging.getLogger()
# client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
client = OpenAI()
comprehension_info = "You will read a sentence, and your task is to answer the comprehension question about that sentence. "
rc_info = "You will read a sentence with a missing word. There are two options for the missing word, and the answer options are 1 or 2. You task is to read the sentence and choose the best option. Please answer with either 1 or 2."

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def get_prediction(prompt, model):
    # prob = []
    prediction = client.chat.completions.create(
        model = model,
        messages = prompt,
        temperature = 0,
        max_tokens = 256,
        logprobs=True,
        top_logprobs=5
    )
    generated_answer = prediction.choices[0].message.content # text
    raw_probs = prediction.choices[0].logprobs.content # probability of all answers

    return generated_answer, raw_probs

# geting response
system_prompt = "You are a helpful assisant. Your task is to follow the instruction and response to the question."

# models to test
# models = ["gpt-3.5-turbo","gpt-4","gpt-4o"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eliciture comprehension and rc attachment")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--task", "-t", type=str, default="comprehension")
    parser.add_argument("--input", "-i", type=str, default="stimuli/comprehension.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="data/")
    args = parser.parse_args()

    prompts = pd.read_csv(args.input, header=0)

    for i, row in tqdm(prompts.iterrows()):
        if args.task == "rc":
            og_prompt = row.prompt
            alt_prompt = row.alt_prompt
            # the order of the two options are randomized for each item (Pezeshkpour & Hruschka, 2023)
            # randomization happens when sending the prompt, so each model might have
            # different models might have different option orders for the same item
            prompt = random.choice([og_prompt, alt_prompt])
            if prompt == og_prompt:
                prompts.loc[i, "answer_option"] = row.option
            else:
                prompts.loc[i, "answer_option"] = row.alt_option
            prompt = rc_info + prompt
        elif args.task == "comprehension":
            prompt = comprehension_info + args.prompt
        else:
            print("wrong task")
            break
        generated_answer, raw_probs = get_prediction(
            prompt=[{"role" : "system", "content": system_prompt},
                    {"role": "user", "content": prompt}],
            model=args.model
        )
        prompts.loc[i, "answer"] = generated_answer.strip()
        prompts.loc[i, "raw_probs"] = str(raw_probs)

        # prompts.to_csv(f"{args.task}_generate_system_{args.model}.csv", index=False)
        output_dir = args.output_dir
        prompts.to_csv(os.path.join(output_dir,f"{args.task}-{args.model}.csv"), index=False)