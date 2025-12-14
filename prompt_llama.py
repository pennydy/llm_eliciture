import torch
from pprint import pprint
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,GPT2Tokenizer, GPT2Model
import argparse
import os
import re
import pandas as pd
from tqdm import tqdm
# from accelerate import disk_offload
# from diffusers import DiffusionPipeline


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

    # system_prompt = "You are a helpful assistant. Your task is to follow the instruction and response to the question."
#     rc_info = "You will read a sentence, and your task is to answer the comprehension question about that sentence. "

#     parser = argparse.ArgumentParser(description="eliciture comprehension")
#     parser.add_argument("--model", "-m", type=str, default="Llama-3.2-1B-Instruct")
#     parser.add_argument("--input", "-i", type=str, default="stimuli/comprehension_nonIC_full_alt.csv")
#     parser.add_argument("--output_dir", "-o", type=str, default="data/")
#     parser.add_argument("--seed", "-s", type=int, default=1)
#     # parser.add_argument("--generate", "-g", type=bool, default=True)
#     args = parser.parse_args()

#     task = re.search(r'\W*/(.*?)\.csv', args.input).group(1)
#     # generate_method=args.generate

#     if "llama" in args.model.lower():
#         model_id = "meta-llama/" + args.model

#         pipe = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         temperature = 0.1
#         )
#     else:
#         print("wrong model")
#         exit()

#     stimuli = pd.read_csv(args.input, header=0)
#     for i, row in tqdm(stimuli.iterrows()):
#         prompt = rc_info + row.prompt

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt},
#         ]

#         outputs = pipe(messags,
#                     max_new_tokens=256,
#                     temperature = 0.1
#         )

#         full_answer = outputs[0]["generated_text"][-1]

#         stimuli.loc[i, "full_answer"] = str(full_answer)
#         stimuli.loc[i, "answer"] = str(full_answer['content'])
    
#         output_dir = os.path.join(args.output_dir, task)
#         if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#         stimuli.to_csv(os.path.join(output_dir,f"{task}_{args.model}_{args.seed}.csv"), index=False)
    
    # # test on a single sentence to see if the method works
    # model_id = "meta-llama/Llama-3.2-1B"
    # pipe = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    # messages = "You will read a sentence, and your task is to answer the comprehension question about that sentence. Sentence: Anna scolded the chef who was routinely letting food go to waste. Question: Why did Anna scold the chef? Answer: "
    # # messages = [
    # #     {"role": "system", "content": "Your task is to follow the instruction and response to the question."},
    # #     {"role": "user", "content": "You will read a sentence, and your task is to answer the comprehension question about that sentence. Sentence: Anna scolded the chef who was routinely letting food go to waste. Question: Why did Anna scold the chef? Answer: "},
    # # ]
    # outputs = pipe(
    #     messages,
    #     max_new_tokens=256,
    #     temperature = 0.1
    # )
    # print(outputs[0])
    # full_answer = outputs[0]["generated_text"][-1]
    # print(full_answer)
    # answer=full_answer['content']
    # print(answer)


    system_prompt = "Your task is to follow the instruction and respond to the question. If you know the answer, let me know what it is. If you don't know the answer, just say so."
    rc_info = "You will read a sentence, and your task is to answer the comprehension question about that sentence. "

    parser = argparse.ArgumentParser(description="eliciture comprehension")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.2-1B-Instruct")
    parser.add_argument("--input", "-i", type=str, default="stimuli/comprehension_nonIC_full_alt.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="data/")
    parser.add_argument("--seed", "-s", type=int, default=1)
    # parser.add_argument("--generate", "-g", type=bool, default=True)
    args = parser.parse_args()

    task = re.search(r'\W*/(.*?)\.csv', args.input).group(1)
    # generate_method=args.generate

    if "llama" in args.model.lower():
        model_id = "meta-llama/" + args.model
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.pad_token_id = model.config.eos_token_id

    else:
        print("wrong model")
        exit()

    stimuli = pd.read_csv(args.input, header=0)
    for i, row in tqdm(stimuli.iterrows()):
        prompt = rc_info + row.prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs=model.generate(tokenized_chat,max_new_tokens=256)
        full_answer = tokenizer.decode(outputs[0],skip_special_tokens=True)

        stimuli.loc[i, "full_answer"] = str(full_answer)
        stimuli.loc[i, "answer"] = str(full_answer.split("\n")[-1])
    
        output_dir = os.path.join(args.output_dir, task)
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        stimuli.to_csv(os.path.join(output_dir,f"{task}_{args.model}_generate_{args.seed}.csv"), index=False)
    
    # # test on a single sentence to see if the method works
    # model_id = "meta-llama/Llama-3.2-3B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id,
    #                                             padding_side="left")
    # tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # # disk_offload(model=model, offload_dir="offload")
    # model.config.pad_token_id = model.config.eos_token_id
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant. Your task is to follow the instruction and response to the question."},
    #     {"role": "user", "content": "Does this sentence explain why Melissa detests the children? If yes, please provide an explanation. If not, just say no and you don't need to provide an explanation. Sentence: Melissa detests the children who are arrogant and rude. Answer: "},
    # ]
    # # messages = "Does this sentence explain why Melissa babysits the children? If yes, please provide an explanation. If not, just say no and you don't need to provide an explanation. Sentence: Melissa babysits the children who are arrogant and rude. Answer: "

    # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # # use **messages if directly passing the string
    # outputs=model.generate(tokenized_chat,max_new_tokens=256)
    # full_answer = tokenizer.decode(outputs[0])
    # print(full_answer)
    # answer = tokenizer.decode(outputs[0],skip_special_tokens=True)
    # print(type(answer))
    # print(answer.split("\n")[-1])

