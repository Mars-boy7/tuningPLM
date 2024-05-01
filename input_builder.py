class PromptBuilder:
    def __init__(self) -> None:
        pass
    
    def build_support_prompt_inputs(
        self, 
        token_list: list,
        head_list: list,
        tail_list: list,
        relation_list: list,
        pid2name: dict,
        mask_token: str,
        rule_file:list,#记得在方法调用给定规则文件的URL
    ):
        relation_name_list = [pid2name[r][0] for r in relation_list]
        #添加转换代码
        rule_list = self.map_rel_rule(rule_file,relation_name_list)
        prompt_input_list = []
        text_list = [' '.join(token) for token in token_list]
        for i in range(len(text_list)):
            #修改relation为rule
            prompt_input = f"{text_list[i]} {head_list[i]} {rule_list[i]} {mask_token} {rule_list[i]} {tail_list[i]}"
            prompt_input_list.append(prompt_input)
        return prompt_input_list

    #根据关系名称获取对应的规则
    def map_rel_rule(self, rule_file, relation_name_list):
        # 创建一个空的列表来存储匹配的规则
        rule_list = []
        
        for one_rule_file in rule_file:
            # 打开规则文件并逐行读取
            with open(one_rule_file, "r") as file:
                next(file)  # 跳过头行
                for line in file:
                    # 移除换行符并按逗号分割
                    parts = line.strip().split(',')
                    # 将字符串表示的列表转换回列表
                    rule = eval(parts[0])
                    label = parts[1].strip()

                    # 检查当前行的标签是否在提供的关系名称列表中
                    if label in relation_name_list:
                        # 如果在，则将整个规则添加到规则列表中
                        rule_list.append(rule)
        
        return rule_list


    #未使用的代码块
    # def build_query_prompt_inputs(
    #     self, 
    #     token_list: list,
    #     head_list: list,
    #     tail_list: list,
    #     mask_token: str,
    # ):
    #     prompt_input_list = []
    #     text_list = [' '.join(token) for token in token_list]
    #     for i in range(len(text_list)):
    #         prompt_input = f"{text_list[i]} {head_list[i]} {mask_token} {tail_list[i]}"
    #         prompt_input_list.append(prompt_input)
    #     return prompt_input_list


class EnsemblePormptBuilder(PromptBuilder):
    def __init__(self) -> None:
        super().__init__()
    
    def build_support_prompt_inputs(
        self, 
        token_list: list, 
        head_list: list, 
        tail_list: list, 
        relation_list: list, 
        pid2name: dict, 
        mask_token: str,
        rule_file:list,#记得在方法调用给定规则文件的URL
    ):
        relation_name_list = [pid2name[r][0] for r in relation_list]
        #添加转换代码
        rule_list = self.map_rel_rule(rule_file,relation_name_list)
        prompt_input_list = []
        text_list = [' '.join(token) for token in token_list]
        for i in range(len(text_list)):
            #修改relation为rule
            prompt_input_1 = f"{text_list[i]} {rule_list[i]} {mask_token} {rule_list[i]} {head_list[i]}  {tail_list[i]}"
            prompt_input_2 = f"{text_list[i]} {head_list[i]} {rule_list[i]} {mask_token} {rule_list[i]} {tail_list[i]}"
            prompt_input_3 = f"{text_list[i]} {head_list[i]} {tail_list[i]} {rule_list[i]} {mask_token} {rule_list[i]}"
            prompt_input_list.extend([prompt_input_1, prompt_input_2, prompt_input_3])
        return prompt_input_list


class EntityMarkerBuilder:
    def __init__(self) -> None:
        pass
    
    def mark(
        self,
        token_list: list,
        head_idx_list: list,
        tail_idx_list: list,
        head_marker_start: str = '<subj>',
        head_marker_end: str = '</subj>',
        tail_marker_start: str = '<obj>',
        tail_marker_end: str = '</obj>',
    ):
        new_token_list = []
        for i in range(len(token_list)):
            tokens = token_list[i]
            h_idx = head_idx_list[i]
            t_idx = tail_idx_list[i]
            
            h_start, h_end = self._get_start_end(h_idx)
            t_start, t_end = self._get_start_end(t_idx)
            
            new_tokens = self._add_marker_to_tokens(
                    h_marker_start=head_marker_start,
                    h_marker_end=head_marker_end,
                    t_marker_start=tail_marker_start,
                    t_marker_end=tail_marker_end,
                    tokens=tokens,
                    h_idx_start=h_start,
                    h_idx_end=h_end,
                    t_idx_start=t_start,
                    t_idx_end=t_end
                )
            new_token_list.append(new_tokens)
        
        return new_token_list
    
    def _get_start_end(self, idx: list):
        # if len(idx) <= 1:
        #     idx_start = idx[0]
        #     idx_end = idx[0]
        # else:
        idx_start = idx[0]
        idx_end = idx[-1]
        return idx_start, idx_end
    
    def _add_marker_to_tokens(
        self, 
        h_marker_start: str,
        h_marker_end: str,
        t_marker_start: str,
        t_marker_end: str,
        tokens: list,
        h_idx_start: int,
        h_idx_end: int,
        t_idx_start: int,
        t_idx_end: int,
    ):  
        if h_idx_start > t_idx_start:
            h_start = tokens[:h_idx_start] + [h_marker_start]
            h_end = h_start + tokens[h_idx_start:h_idx_end+1] + [h_marker_end]
            t_start = h_end + tokens[h_idx_end+1:t_idx_start] + [t_marker_start]
            t_end = t_start + tokens[t_idx_start:t_idx_end+1] + [t_marker_end]
            new_tokens = t_end + tokens[t_idx_end+1:]
        else:
            t_start = tokens[:t_idx_start] + [t_marker_start]
            t_end = t_start + tokens[t_idx_start:t_idx_end+1] + [t_marker_end]
            h_start = t_end + tokens[t_idx_end+1:h_idx_start] + [h_marker_start]
            h_end = h_start + tokens[h_idx_start:h_idx_end+1] + [h_marker_end]
            new_tokens = h_end + tokens[h_idx_end+1:]
        return new_tokens


class InputFeatureBuilder:
    def __init__(self) -> None:
        pass
    
    def build_input_feature(
        self,
        tokenizer,
        text_list: list,
        max_len: int = 256
    ):
        return tokenizer(
            text=text_list,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )