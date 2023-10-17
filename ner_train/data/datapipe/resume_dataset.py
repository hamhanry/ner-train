import torch
from torch.utils.data import Dataset

class ResumeDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.resume = self.data['resume']
        self.targets = self.data['list']
        self.max_len = max_len

    def __len__(self):
        return len(self.resume)

    def __getitem__(self, index):
        resume = str(self.resume[index])
        resume = " ".join(resume.split())

        inputs = self.tokenizer.encode_plus(
            resume,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }