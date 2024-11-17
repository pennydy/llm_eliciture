import numpy as np
import pandas as pd
import logging
import random
import re
from openai import OpenAI
from tqdm import tqdm
import argparse
import os

logger = logging.getLogger()
# client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
client = OpenAI()
comprehension_info = "You will read a sentence, and your task is to answer the comprehension question about that sentence. "
rc_info = "You will read a sentence where the part after \"who\" is missing. There are two options for how to start the missing part, and the answer options are 1 or 2. You task is to read the sentence and choose the best option. Please answer with either 1 or 2."
# rc_info = "You will read a sentence with a missing word. There are two options for the missing word, and the answer options are 1 or 2. You task is to read the sentence and choose the best option. Please answer with either 1 or 2."
pronoun_info = "You will read a sentence, and your task is to write a follow-up sentence. The two people mentioned in the first sentence have the same gender, and the gender is marked with (m) if they are male and with (f) if they are female. Please complete the follow-up sentence by avoiding humor. "
pronoun_pro_info = "You will read a sentence, and your task is to write a follow-up sentence. The two people mentioned in the first sentence have the same gender, and the gender is marked with (m) if they are male and with (f) if they are female. Please complete the follow-up sentence after the pronoun by avoiding humor. "
# pronoun_info = "You will read a sentence, and your task is to write a follow-up sentence. The provided sentence will involve two people with the same gender, either both male or both female. Please only write the follow-up sentence by avoiding humor. "
# pronoun_pro_info = "You will read a sentence, and your task is to write a follow-up sentence after the provided pronoun. The people mentioned in the provided sentence have the same gender, either both male or both female. Please complete the follow-up sentence after the provided pronoun by avoiding humor. "

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def get_prediction(prompt, model, seed):
    # prob = []
    prediction = client.chat.completions.create(
        model = model,
        messages = prompt,
        # seed = seed,
        temperature = 0,
        max_tokens = 256,
        logprobs=True,
        top_logprobs=5 # ranging from 0 to 20, the number of most likely tokens to return at each token position
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
    # parser.add_argument("--task", "-t", type=str, default="comprehension")
    parser.add_argument("--input", "-i", type=str, default="stimuli/comprehension.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="data/")
    parser.add_argument("--seed", "-s", type=int, default=1)
    args = parser.parse_args()

    prompts = pd.read_csv(args.input, header=0)
    task = re.search(r'\W*/(.*?)\.csv', args.input).group(1)

    for i, row in tqdm(prompts.iterrows()):
        if "rc" in task:
            # randomization (Pezeshkpour & Hruschka, 2023)
            if "answer_option" in prompts.columns:
                prompt = row.prompt
            else:
                # the order of the two options are randomized for each item 
                # randomization happens when sending the prompt, so each model might have
                # different models might have different option orders for the same item
                og_prompt = row.prompt_1
                alt_prompt = row.prompt_2
                prompt = random.choice([og_prompt, alt_prompt])
                if prompt == og_prompt:
                    prompts.loc[i, "answer_option"] = row.option_1
                elif prompt == alt_prompt:
                    prompts.loc[i, "answer_option"] = row.option_2
                else:
                    print("wrong rc prompt!")
            prompt = rc_info + prompt
        elif "pronoun" in task:
            if "pronoun_free" in task:
                prompt = pronoun_info + row.prompt
            elif "pronoun_pro" in task:
                prompt = pronoun_pro_info + row.prompt
            else: 
                print("wrong pronoun prompt!")
                break
        elif  "comprehension" in task:
            prompt = comprehension_info + row.prompt
        else:
            print("wrong task")
            break
        generated_answer, raw_probs = get_prediction(
            prompt=[{"role" : "system", "content": system_prompt},
                    {"role": "user", "content": prompt}],
            seed=args.seed,
            model=args.model
        )
        prompts.loc[i, "answer"] = generated_answer.strip()
        prompts.loc[i, "raw_probs"] = str(raw_probs)

        output_dir = os.path.join(args.output_dir, task)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # prompts.to_csv(f"{args.task}_generate_system_{args.model}.csv", index=False)
        # output_dir = args.output_dir
        prompts.to_csv(os.path.join(output_dir,f"{task}-{args.model}_{args.seed}.csv"), index=False)