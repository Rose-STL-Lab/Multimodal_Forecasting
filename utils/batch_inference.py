import random
from tqdm import tqdm
from loguru import logger
from utils.modelchat import LLMChatModel

def create_batched(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]

def create_batch_prompt(data, case):
    prompt = []
    cur_idx = []
    if case in [1]:
        input_column = 'input_text'
    elif case in [2, 3, 4]:
        input_column = 'input_text_time'
    instruction_column = f'instruction-{case}'

    for index, row in data.iterrows():
        content = [{"role": "system", "content": "You are a weather forecast assistant. Given the weather data from a recent day, predict the weather conditions for the following days"}, 
                   {"role": "user", "content": row[instruction_column] + row[input_column]}]
        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx

def create_batch_prompt_in_Context(data, data_train, case):
    prompt = []
    cur_idx = []
    if case == 1:
        input_column = 'input_text'
        output_column = 'output_text'
    elif case == 2:
        input_column = 'input_text_time'
        output_column = 'output_text_time'
    elif case == 3:
        input_column = 'input_text_time'
        output_column = 'output_text'
    elif case == 4:
        input_column = 'input_text_time'
        output_column = 'output_time'
    instruction_column = f'instruction-{case}'
    
    randaom_loc = random.randint(0, len(data_train) - 1)
    example_input = data_train.iloc[randaom_loc][input_column]
    example_output = data_train.iloc[randaom_loc][output_column]

    for index, row in data.iterrows():
        content = [{"role": "system", "content": "You are a weather forecast assistant. Given the weather data from a recent day, predict the weather conditions for the following days"}, 
                   {"role": "user", "content": row[instruction_column] + example_input}, 
                   {"role": "assistant", "content": example_output}, 
                   {"role": "user", "content": row[instruction_column] + row[input_column]}]
        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx

def batch_inference(
    data,
    case,
    log_path,
    device,
    token,
    dataset,
    window_size,
    finetuned
):
    batches = list(create_batched(data, 8))
    logger.remove()
    logger.add(log_path, rotation="10 MB", mode="w")
    data['idx'] = data.index
    results = [{"pred_output": "Wrong output format"} for _ in range(len(data))]
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset, finetuned, case, device, window_size)
    
    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt(batch, case)
        output_texts = model_chat.chat(prompt)
        for index, output_text in enumerate(output_texts):
            response = output_text.split("assistant")[-1]
            logger.info("Content for row: " + str(cur_idx[index]) + " Content: " + prompt[index][-1]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) +  " Content: " + response)
            results[cur_idx[index]] = (
                    {"pred_output": response})
    
    return results

def batch_inference_inContext(
    data,
    data_train,
    case,
    log_path,
    device,
    token,
    dataset,
    window_size,
    finetuned
):
    batches = list(create_batched(data, 3))
    logger.remove()
    logger.add(log_path, rotation="10 MB", mode="w")
    data['idx'] = data.index
    results = [{"pred_output": "Wrong output format"} for _ in range(len(data))]
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset, finetuned, case, device, window_size)

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt_in_Context(batch, data_train, case)
        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split("assistant")[-1]
            logger.info("Content for row: " + str(cur_idx[index]) + " Content: " + prompt[index][-1]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) +  " Content: " + response)
            results[cur_idx[index]] = (
                    {"pred_output": response})
    
    return results