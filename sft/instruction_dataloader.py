from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch
import torch.nn.functional as F

class CompressDataset(IterableDataset):
    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        self.examples = examples
        self.batch_size = batch_size


    def __iter__(self):

        MAX_LEN = 510   # 你要固定长度

        batch = {key: [] for key in self.examples[0].keys()}
        count = 0

        def pad_and_stack(values, pad_val):

            padded = []

            for v in values:

                # ===== 截断 =====
                if v.size(0) > MAX_LEN:
                    v = v[:MAX_LEN]

                # ===== padding =====
                elif v.size(0) < MAX_LEN:
                    v = F.pad(v, (0, MAX_LEN - v.size(0)), value=pad_val)

                padded.append(v)

            return torch.stack(padded)

        for example in self.examples:

            for key, value in example.items():
                batch[key].append(value)

            count += 1

            # ===== 满 batch =====
            if count == self.batch_size:

                output = {}

                for key, value in batch.items():

                    # None 类型直接保留
                    if value[0] is None:
                        output[key] = None
                        continue

                    # ⭐⭐⭐ 关键：不同 key 不同 padding
                    if key == "input_ids":
                        output[key] = pad_and_stack(value, 0)

                    elif key == "lm_targets":
                        output[key] = pad_and_stack(value, 0)

                    else:
                        # 其他 tensor 默认 stack
                        output[key] = torch.stack(value)

                yield output

                # reset
                batch = {key: [] for key in batch}
                count = 0
    
    def pad_to_fixed_len(self, inputs, max_len=510, pad_token_id=0):
        input_ids = inputs["input_ids"]
        lm_targets = inputs["lm_targets"]

        # 当前长度
        cur_len = input_ids.size(1)

        # ====== 截断 ======
        if cur_len > max_len:
            input_ids = input_ids[:, :max_len]
            lm_targets = lm_targets[:, :max_len]

        # ====== padding ======
        elif cur_len < max_len:
            pad_len = max_len - cur_len

            input_ids = F.pad(
                input_ids,
                (0, pad_len),
                value=-100
            )

            lm_targets = F.pad(
                lm_targets,
                (0, pad_len),
                value=-100  # VERY IMPORTANT
            )

        inputs["input_ids"] = input_ids
        inputs["lm_targets"] = lm_targets
        return inputs

                
                
def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)
    
    raise Exception("Don't exist [{task_type}] task.")