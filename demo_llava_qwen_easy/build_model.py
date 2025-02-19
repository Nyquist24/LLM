from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
from transformers import LlavaForConditionalGeneration,LlavaConfig


clip_model_name_or_path = ("/home1/dongyh/openai/clip-vit-large-patch14-336")
qwen_model_name_or_path = "/home1/dongyh/Qwen1.5-4B-Chat"
modify_qwen_tokenizer_dir = "/home1/dongyh/Qwen1.5-4B-Chat"
modify_qwen_tokenizer = AutoTokenizer.from_pretrained(modify_qwen_tokenizer_dir)

print(f'<image> token of Qwen is:',modify_qwen_tokenizer.encode("<image>"))



clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map="cuda:0")
llm_model = AutoModelForCausalLM.from_pretrained(qwen_model_name_or_path, device_map="cuda:0")

llm_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)

# Initializing a CLIP-vision config
vision_config = clip_model.vision_model.config

# Initializing a Llama config
text_config = llm_model.config

# Initializing a Llava llava-1.5-7b style configuration
configuration = LlavaConfig(vision_config, text_config)

# Initializing a model from the llava-1.5-7b style configuration
model = LlavaForConditionalGeneration(configuration)


model.vision_tower.vision_model = clip_model.vision_model
model.language_model = llm_model

model.config.pad_token_id = llm_tokenizer.pad_token_id
model.config.image_token_index = llm_tokenizer.encode("<image>")[0]

model.save_pretrained("build_llava/qwen")
llm_tokenizer.save_pretrained("build_llava/qwen")

autoprocessor = AutoProcessor.from_pretrained(clip_model_name_or_path)
autoprocessor.save_pretrained("build_llava/clip")

# 下面的代码是加载并验证效果的
# from transformers import LlavaProcessor, LlavaForConditionalGeneration
# import torch


# model_name_or_path = "show_model/model001"  # 
# # model_name_or_path = "test_model_copy/model001"  #

# llava_processor = LlavaProcessor.from_pretrained(model_name_or_path)
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_name_or_path, device_map="cuda:5", torch_dtype=torch.bfloat16
# )

# from PIL import Image

# prompt_text = "<image>\nWhat are these?"


# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt_text},
# ]
# prompt = llava_processor.tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )


# image_path = "000000039769.jpg"
# image = Image.open(image_path)


# inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

# for tk in inputs.keys():
#     inputs[tk] = inputs[tk].to(model.device)
# generate_ids = model.generate(**inputs, max_new_tokens=20)
# gen_text = llava_processor.batch_decode(
#     generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
# )[0]

# print(gen_text)