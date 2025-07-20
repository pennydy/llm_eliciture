import torch
from pprint import pprint
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import argparse
import os
import re
import pandas as pd
from tqdm import tqdm


# get the log prob of each word in one sentence
# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/15
def to_tokens_and_logprobs(model_id, model, tokenizer, input_texts):
    # pad the sentence with a BOS token for gpt2 and pythia
    if model_id == "gpt2" or "pythia" in model_id:    
        bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.pad_token
        padded_texts = [bos_token + " " + text for text in input_texts]
    # BOS handled by the tokenizer of llama models
    else:
        padded_texts = input_texts

    input_ids = tokenizer(padded_texts, padding=True, return_tensors="pt").input_ids
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

    parser = argparse.ArgumentParser(description="eliciture comprehension and rc attachment")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.2-1B")
    parser.add_argument("--input", "-i", type=str, default="stimuli/rc_sent.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="data/")
    parser.add_argument("--seed", "-s", type=int, default=1)
    parser.add_argument("--range", "-r", type=int, default=None)
    parser.add_argument("--version", "-v", type=str, default=None) # for pythia models
    args = parser.parse_args()

    task = re.search(r'\W*/(.*?)\.csv', args.input).group(1)
    
    if "llama" in args.model.lower():
        # model_id = ["Llama-3.2-3B", "Llama-3.2-1B","Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "gpt2"]
        model_id = "meta-llama/" + args.model
    elif args.model == "gpt2":
        model_id = "gpt2"
    elif "pythia" in args.model:
        model_id = "EleutherAI/" + args.model
        if args.version is None:
            print("provide a version of the pythia model!")
        else:
            pythia_version = args.version
        
    else:
        print("check model id!")

    if "pythia" in args.model:
                
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=pythia_version,
            cache_dir="./cache/"+args.model+"/"+pythia_version,
            padding_side="left"
        )
        # tokenizer.pad_token = tokenizer.eos_token

        model = GPTNeoXForCausalLM.from_pretrained(
            model_id,
            revision=pythia_version,
            cache_dir="./cache/"+args.model+"/"+pythia_version,
        )
        
        # model.config.pad_token_id = model.config.eos_token_id


    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                    padding_side="left")
        # tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        # model.config.pad_token_id = model.config.eos_token_id
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    stimuli = pd.read_csv(args.input, header=0)
    task = re.search(r'\W*/(.*?)\.csv', args.input).group(1)
    for i, row in tqdm(stimuli.iterrows()):
        sentence = row.sentence
        sentence = [sentence]
        critical_logprobs_sum = 0
        logprobs = to_tokens_and_logprobs(model_id, model, tokenizer, sentence)
        # for ", and I don't know why." continuation, range=8
        # for " and so" continuation, range=2
        # for " because" continuation, range=1
        if "nonIC" in task and args.range is not None:
            for n in range(0-args.range,0):
                critical_logprobs_sum += logprobs[n][1]
            sentence_logprobs = critical_logprobs_sum/args.range
            stimuli.loc[i,"critical_region"] = str(logprobs[-args.range:])
            stimuli.loc[i,"critical_region_sum"] = float(critical_logprobs_sum)
            stimuli.loc[i,"critical_region_logprob"] = float(sentence_logprobs)
            # # "why" is the penultimate token in the continuation
            # stimuli.loc[i,"critical_token"] = str(logprobs[-2][0])
            # stimuli.loc[i,"critical_prob"] = float(logprobs[-2][1])
        elif "comprehension_sent" in task or "nonIC" in task or "rc" in task:
            stimuli.loc[i,"critical_token"] = str(logprobs[-1][0])
            stimuli.loc[i,"critical_prob"] = float(logprobs[-1][1])
        else:
            print("wrong task!")

        stimuli.loc[i,"logprob"] = str(logprobs).strip()

    output_dir = os.path.join(args.output_dir, task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stimuli.to_csv(os.path.join(output_dir,f"{task}_{args.model}_{args.seed}.csv"), index=False)