import torch
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate:int, pretrained_name:str, num_classes:int):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(pretrained_name)
        self.l2 = torch.nn.Dropout(drop_rate)
        self.l3 = torch.nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output