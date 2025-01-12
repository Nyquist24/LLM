from dataclasses import dataclass
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

# 数据集所在的路径
data_dir = "../data/LLaVA-CC3M-Pretrain-595K"

# 数据集构造：1.__len__(); 2.__getitem__()
class LlavaDataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset = dataset_dir
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)
        
    # 构建数据集，读取chat.json和图像文件夹路径
    def build_dataset(self, data_dir):
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")  # 聊天数据的JSON文件
        image__dir = data_dir.joinpath("images_dl")  # 图像文件夹路径
        
        # 读取chat.json文件并转化为字典列表
        chat_data = pd.read_json(chat_file).to_dict(orient='records')
        
        return chat_data, image__dir
    
    # 获取数据集大小
    def __len__(self):
        return len(self.chat_data)
    
    # 获取单条数据
    def __getitem__(self, index):
        cur_data = self.chat_data[index]
        
        # 获取人类输入和GPT的输出
        human_input = cur_data['conversations'][0]['value']
        gpt_output = cur_data['conversations'][1]['value']
        
        # 获取对应的图像路径
        image_path = self.image_dir.joinpath(cur_data['image'])
        
        return human_input, gpt_output, image_path

# 初始化数据集对象
test_llavadataset = LlavaDataset(dataset_dir=data_dir)


# 初始化一个processor
from transformers import AutoProcessor, LlavaForConditionalGeneration

llava_model_name_path = '/home/fangyf/dongyh/train_llava/show_model/model001'

# 加载处理器和模型
llava_processor = AutoProcessor.from_pretrained(llava_model_name_path)
llava_model = LlavaForConditionalGeneration.from_pretrained(llava_model_name_path, torch_dtype=torch.bfloat16, device_map='cuda:0')

# 选取数据集中的一条数据
test12345 = test_llavadataset[12345]

# 定义一个数据输出类
@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor
    
# 构建图像-问答输入输出的函数
def build_qaimage(processor, q_text, a_text, image_path):
    # 设置消息格式，给出系统和用户的输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    
    # 使用处理器的tokenizer将文本转化为模型的输入格式
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 加载图像并通过处理器进行转换
    raw_image = Image.open(image_path)
    
    # 处理图像和问题文本
    inputs = processor(prompt, raw_image, return_tensors='pt')
    
    # 处理答案文本
    a_inputs_ids = processor.tokenizer(a_text, return_tensors='pt', padding="longest", truncation=True)
    
    return QaImageOutput(
        q_input_ids=inputs['input_ids'],
        pixel_values=inputs['pixel_values'],
        a_input_ids=a_inputs_ids['input_ids']
    )

# 生成一个示例的图像-问答输入输出
c = build_qaimage(processor=llava_processor, q_text=test12345[0], a_text=test12345[1], image_path=test12345[2])
print(c.q_input_ids, c.a_input_ids)


# 定义批处理的collator类
class TrainLLavaModelCollator:
    def __init__(self, processor, IGNORE_INDEX):
        self.processor = processor
        self.ignore_index = IGNORE_INDEX
    
    # 转化单个数据样本为模型输入和标签
    def convert_one_piece(self, q_input_ids, a_input_ids):
        # 拼接问题和答案的input_ids，并添加结束符
        inputs_ids = torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        
        # 构建标签，问题部分使用忽略标签
        labels = torch.concat([
            torch.full_like(input=q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
        ], axis=1)
        
        
        
        return inputs_ids, labels
    
    # 批处理多个数据样本
    def __call__(self, features, *args, **kwds):
        input_ids_list = []
        labels_list = []
        pixels_list = []
        max_input_len_list = []
        
        # 处理每个样本
        for feature in features:
            qaimage_output = build_qaimage(
                processor=self.processor,
                q_text=feature[0], a_text=feature[1], image_path=feature[2]
            )
            # 获取单个样本的输入ids和标签
            temp_input_ids, temp_labels = self.convert_one_piece(
                q_input_ids=qaimage_output.q_input_ids,
                a_input_ids=qaimage_output.a_input_ids
            )
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixels_list.append(qaimage_output.pixel_values)
            max_input_len_list.append(temp_input_ids.shape[1])
        
        # 对同一批次的数据统一长度
        max_input_len = max(max_input_len_list)
        
        # 拼接输入ids，进行填充
        final_input_ids = torch.cat([
            torch.cat([
                torch.full((1, max_input_len - max_input_len_list[index]), fill_value=self.processor.tokenizer.pad_token_id), value
            ], dim=1)
            for index, value in enumerate(input_ids_list)
        ])
        
        # 拼接标签，进行填充
        final_labels = torch.concat(
            [torch.concat([torch.full((1, max_input_len - max_input_len_list[index]), fill_value=self.ignore_index), value], axis=1)
             for index, value in enumerate(labels_list)], axis=0)
        
        # 拼接图像像素值
        final_pixels_values = torch.concat(pixels_list, axis=0)
        
        
        # Qwen采用的是后填充，前填充和后填充的区别在于位置编码，不清楚使用attention mask之后效果是否是一致的
        # final_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        # final_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index)
        # final_pixels_values = torch.stack(pixels_list)
        
        
        
        # 构建attention mask
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_ids] = 0
        
        return {
            'input_ids': final_input_ids,
            'labels': final_labels,
            'pixel_values': final_pixels_values,
            'attention_mask': attention_mask
        }

# 初始化数据批处理collator
tlmc = TrainLLavaModelCollator(processor=llava_processor, IGNORE_INDEX=-100)

# 使用collator处理一个批次的数据
d = tlmc([test_llavadataset[1]])

# 将数据移动到模型所在的设备上
for tk in d.keys():
    d[tk] = d[tk].to(llava_model.device)
    
# 使用LLaVA模型进行推理或训练
model_out = llava_model(**d)
print(model_out)
