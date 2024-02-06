import torch
import re
import math
import string


MODEL2BRACKET_IDS = {
    "m2m": {
        '[': [542],  # , 1448],
        ']': [11355]  # , 494]
    },
    'nllb': {
        '[': [709],  # 248415],  # _[, [
        ']': [10109],  # , 248414]  # _] , ]
        '(': [104],
        ')': [14229]
    },
    'mbart': {
        '[': [378],
        ']': [10114]
    }
}

BRACKET_IDS = {709, 248415, 10109, 248414}
PUNC_LIST = set(string.punctuation)
CHINESE_PUNC = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'
PUNC_LIST.remove("\'")


class Candidate:
    def __init__(self, text_ids, score, accumulate_scores=None):
        self.text_ids = text_ids
        self.score = score
        self.flag = False
        self.count = 0.
        self.max_position = 0
        self.accumulate_scores = accumulate_scores
        self.search_tree = None
        self.close_bracket_position = -1
        self.open_bracket_position = -1
        self.min_heap = []

    def to_cuda(self, device):
        if self.text_ids is not None:
            self.text_ids = self.text_ids.to(device)
        if self.score is not None:
            self.score = self.score.to(device)

    def update_smallest_candidate(self):
        self.score = self.min_heap[0][0]
        self.text_ids = self.min_heap[0][2]
        self.accumulate_scores = self.min_heap[0][3]


class Node:
    def __init__(self, text=None, log_prob=None, acc_log_prob=None, upperbound=None):
        self.text = text
        self.log_prob = log_prob
        self.acc_log_prob = acc_log_prob
        self.child = []
        self.level = 0
        self.upperbound = upperbound

    def add_child_node(self, node):
        self.child.append(node)
        node.level = self.level + 1

    def __repr__(self):
        repr_str = "{} - {}: {}/{}".format(self.level, self.text, round(self.log_prob, 2), round(self.acc_log_prob, 2))
        return repr_str


def print_tree(root_node):
    if not root_node:
        return
    print(root_node)
    for node in root_node.child:
        print_tree(node)


def check_punctuation(text):
    return text in CHINESE_PUNC or text in PUNC_LIST


def preprocess(mt, md, text, is_tokenized=False):
    if not is_tokenized:
        tokenized_en_text = mt.tokenize(text)
    else:
        tokenized_en_text = text
    tmp = [md.tokenize([_]) for _ in tokenized_en_text]
    return ' '.join(tmp)


def preprocess2(text, org_text, mt, md, is_tokenized=False, lang=None):
    if not is_tokenized:
        tokenized_en_text = mt.tokenize(text)
    else:
        tokenized_en_text = text
    org_text = re.sub(" +", " ", org_text)
    parts = [md.detokenize([_]) for _ in tokenized_en_text]
    p1 = 0
    tokenized_parts = []
    for part in parts:
        if org_text[p1:].startswith(part):
            if part in PUNC_LIST:
                tokenized_parts.append(f" {part} ")
            else:
                tokenized_parts.append(org_text[p1:p1 + len(part)])
            p1 = p1 + len(part)
        elif org_text[p1 + 1:].startswith(part):
            if part in PUNC_LIST:
                tokenized_parts.append(f" {part} ")
            else:
                tokenized_parts.append(org_text[p1:p1 + 1 + len(part)])
            p1 = p1 + 1 + len(part)
        else:
            print(text)
            print(org_text)
            print("Fail")
            break
    result = ''.join(tokenized_parts)
    result = re.sub(" +", " ", result)
    return result.strip()


def post_process(text, org_text):
    x_pointer = 0
    org_x_pointer = 0
    new_str = []
    if org_text.strip() == '':
        return org_text
    while x_pointer < len(text) or org_x_pointer < len(org_text):
        if org_x_pointer == len(org_text):
            assert text[x_pointer] in [' ', '[', ']']
            new_str.append(text[x_pointer])
            x_pointer += 1
            continue
        if x_pointer == len(text):
            new_str.append(org_text[org_x_pointer])
            org_x_pointer += 1
            continue
        if org_text[org_x_pointer] == text[x_pointer]:
            new_str.append(org_text[org_x_pointer])
            x_pointer += 1
            org_x_pointer += 1
        else:
            if text[x_pointer] == '[' or text[x_pointer] == ']':
                new_str.append(text[x_pointer])
                x_pointer += 1
            else:
                assert org_text[org_x_pointer] == ' ' or text[x_pointer] == ' ', f"\n {org_text} \n {text}"
                if text[x_pointer] == ' ' and org_text[org_x_pointer] != ' ':
                    x_pointer += 1
                elif text[x_pointer] != ' ' and org_text[org_x_pointer] == ' ':
                    new_str.append(org_text[org_x_pointer])
                    org_x_pointer += 1
    return ''.join(new_str)


def tokenize_non_whitespace(template_text, tokenizer):
    tmp = []
    for i, token in enumerate(template_text.split(' ')):
        if i == 0:
            tmp.extend(tokenizer.tokenize(token))
        else:
            tmp1 = [_ for _ in tokenizer.tokenize(token) if _ != '▁']
            tmp.extend([_.replace('▁', '') for _ in tmp1])
    template_ids = [tokenizer.convert_tokens_to_ids(_) for _ in tmp]
    return template_ids


def compute_number_combination(n, num_brackets, num_choices_per_bracket=2):
    return math.comb(n + num_brackets, num_brackets) * (num_choices_per_bracket**num_brackets)


def sent_scoring(model, tokenizer, org_text, target_text, cuda):
    input_text = f'{target_text}{tokenizer.eos_token}'
    input_ids = torch.LongTensor(tokenizer.encode(input_text)).unsqueeze(0)  # Batch size 1
    org_input_ids = torch.LongTensor(tokenizer.encode(org_text))
    tgt_length = input_ids.shape[1] - org_input_ids.shape[0]
    print(tgt_length)
    target_ids = input_ids.clone()
    target_ids[:, :-tgt_length] = -100
    print(input_ids)
    print(target_ids)
    if cuda:
        input_ids = input_ids.to('cuda')
        target_ids = target_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    sentence_prob = outputs.loss.item()*tgt_length
    return sentence_prob


@torch.no_grad()
def enc_dec_scoring(input_ids, target_ids, model, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=target_ids[:, 1:].contiguous())
    # sentence_prob = outputs.loss.item()
    labels = target_ids[:, 2:].contiguous()
    logits = outputs.logits[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    mask_lm_loss = []
    for i in range(labels.shape[0]):
        loss = loss_fct(logits[i].reshape(-1, model.config.vocab_size), labels[i])
        length = (labels[i] != -100).sum()
        loss = loss[:length]
        loss = torch.cumsum(loss, dim=0)
        mask_lm_loss.append(loss.cpu())
    return mask_lm_loss
