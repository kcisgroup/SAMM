import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from model.disencoding import DisEncoding
class Mining(nn.Module):
    def __init__(self, fea_emb, num_emb, fea_num, lm_path, h_dim, n_res_layers, device):
        super(Mining, self).__init__()
        self.fea_emb = fea_emb
        self.num_emb = num_emb
        self.fea_num = fea_num
        self.lm_path = lm_path
        self.h_dim = h_dim
        self.n_res_layers = n_res_layers
        self.device = device
        self.disencoding = DisEncoding(self.fea_emb, self.num_emb, self.fea_num, self.h_dim, self.n_res_layers)
        self.MLM = AutoModelForMaskedLM.from_pretrained(lm_path)

    def forward(self, x):

        output, recons_loss, encoding_one_hot = self.disencoding(x)
        encoding_one_hot = encoding_one_hot.to(torch.int)

        sen_lists, custom_vocab = self.convert_encoding_to_sens(encoding_one_hot)

        tokenizer = AutoTokenizer.from_pretrained(self.lm_path)

        vocab_dict = self.convert_custom_vocab_to_vocab_dict(custom_vocab, tokenizer)
        sen_ids = self.convert_sens_to_ids(sen_lists, tokenizer)

        inputs = self.mask_ids(sen_ids, vocab_dict, tokenizer, max_length=self.fea_num+2)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.MLM.resize_token_embeddings(len(tokenizer))  # 调整词嵌入层大小以适应新词汇表
        outputs = self.MLM(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_states = hidden_states[-1]

        batch_size, seq, dim = last_hidden_states.shape
        cls_vectors = last_hidden_states.view((batch_size, -1))
        mlm_loss = outputs.loss

        return encoding_one_hot, recons_loss, mlm_loss, outputs, sen_ids, cls_vectors


    def convert_sens_to_str(self, sen_ids):
        sen_strs = []
        for i in range(len(sen_ids)):
            sen_str = ""
            for j in sen_ids[i]:
                sen_str += str(j)
            sen_strs.append(sen_str)
        return sen_strs

    def convert_encoding_to_sens(self, encoding_one_hots):
        sen_lists = []
        encoding_one_hots = encoding_one_hots.tolist()
        custom_vocab = []
        for i in range(len(encoding_one_hots)):
            sen = []
            for j in range(len(encoding_one_hots[0])):
                token = ""
                token += str(j)
                for num in encoding_one_hots[i][j]:
                    token += str(num)
                if token not in custom_vocab:
                    custom_vocab.append(token)
                sen.append(token)
            sen_lists.append(sen)
        return sen_lists, custom_vocab

    def convert_custom_vocab_to_vocab_dict(self, custom_vocab, tokenizer):
        vocab_dict = {}
        for i in custom_vocab:
            vocab_dict[i] = tokenizer.convert_tokens_to_ids(i)
        vocab_dict['[MASK]'] = tokenizer.convert_tokens_to_ids('[MASK]')
        return vocab_dict

    def convert_sens_to_ids(self, sens, tokenizer):
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")  # 获取[CLS]的ID
        sep_id = tokenizer.convert_tokens_to_ids("[SEP]")  # 获取[SEP]的ID
        input_ids = []
        for i in range(len(sens)):
            one_ids = [cls_id]
            for j in sens[i]:
                one_ids.append(tokenizer.convert_tokens_to_ids(j))
            one_ids.append(sep_id)
            input_ids.append(one_ids)
        return input_ids

    def mask_ids(self, sens_ids, vocab_dict, tokenizer, max_length=60, mask_prob=0.15):
        input_ids = sens_ids
        for i in range(len(input_ids)):
            input_ids[i] = self.pad_and_truncate(input_ids[i], max_length)

        input_ids = torch.tensor(input_ids)
        batch, seq_len = input_ids.shape
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        labels = input_ids.clone()
        masked_indices = torch.zeros_like(input_ids)

        for i in range(batch):
            for j in range(seq_len):
                if input_ids[i, j] != tokenizer.pad_token_id:
                    if torch.rand(1).item() < mask_prob:
                        masked_indices[i, j] = 1
                        labels[i, j] = -100


                        prob = torch.tensor([0.8, 0.2])
                        choice = torch.multinomial(prob, num_samples=1).item()
                        if choice == 1:
                            input_ids[i, j] = vocab_dict['[MASK]']

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        return inputs

    def pad_and_truncate(self, sequence, max_length):
        sequence = list(sequence)
        while len(sequence) < max_length:
            sequence.append(0)  # 添加填充 token
        if len(sequence) > max_length:
            sequence = sequence[:max_length]  # 截断过长的部分
        return sequence



if __name__ == "__main__":
    pass

