import torch
from pprint import pprint
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,GPT2Tokenizer, GPT2Model
import argparse
import os
import re
import pandas as pd
from tqdm import tqdm


# get the log prob of each word in one sentence
def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    text_sequence = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
    return text_sequence


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eliciture rc attachment")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.2-1B")
    parser.add_argument("--input", "-i", type=str, default="stimuli/rc_sent.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="data/")
    parser.add_argument("--seed", "-s", type=int, default=1)
    args = parser.parse_args()

    # sentence = "Anna scolded the chef of the aristocrats who were"
    # logprobs = to_tokens_and_logprobs(model, tokenizer, sentence)
    # print(logprobs)

    if "llama" in args.model.lower():
        # model_id = ["Llama3.2-3B", "Llama3.2-1B"]
        model_id = "meta-llama/" + args.model
    elif args.model == "gpt2":
        model_id = "gpt2"
        # model = GPT2Model.from_pretrained('gpt2')
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
        #                                           padding_side="left")

    else:
        print("check which model!")

    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.pad_token_id = model.config.eos_token_id

    stimuli = pd.read_csv(args.input, header=0)
    task = re.search(r'\W*/(.*?)\.csv', args.input).group(1)
    for i, row in tqdm(stimuli.iterrows()):
        sentence = row.sentence

        logprobs = to_tokens_and_logprobs(model, tokenizer, sentence)

        stimuli.loc[i,"critical_token"] = str(logprobs[-1][0])
        stimuli.loc[i,"critical_prob"] = float(logprobs[-1][1])
        stimuli.loc[i,"logprob"] = str(logprobs).strip()

        output_dir = os.path.join(args.output_dir, task)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stimuli.to_csv(os.path.join(output_dir,f"{task}_{args.model}_{args.seed}.csv"), index=False)