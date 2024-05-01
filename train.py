from sampler import RelationSampler, ExampleSampler
from input_builder import PromptBuilder, EntityMarkerBuilder, InputFeatureBuilder
from transformers import PreTrainedTokenizer
from model.proto import ProtoFSRE_Entity
import torch
import numpy as np
import random
import time
import os
from process import gen_rule_set



class ProtoEntityTrainer:
    def __init__(
        self,
        train_data: dict,
        val_data: dict,
        pid2name:dict,
        tokenizer: PreTrainedTokenizer,
        train_N: int,
        train_K: int,
        train_q: int,
        epoch: int,
        batch: int,
        val_N: int,
        val_K: int,
        val_q: int,
        val_batch: int,
    ) -> None:
        self.pid2name = pid2name
        self.tokenizer = tokenizer
        self.special_token_ids = self._init_tokenizer()
        
        self.train_data = train_data
        self.train_N = train_N
        self.train_K = train_K
        self.train_q = train_q
        self.rs_train = RelationSampler(
            FewRel_data=self.train_data,
            N=self.train_N
        )
        self.es_train = ExampleSampler(
            FewRel_data=self.train_data,
            K=self.train_K,
            q=self.train_q
        )
        self.epoch = epoch
        self.batch = batch
        
        self.val_data = val_data
        self.val_N = val_N
        self.val_K = val_K
        self.val_q = val_q
        self.val_batch = val_batch
        self.rs_val = RelationSampler(
            FewRel_data=self.val_data,
            N=self.val_N
        )
        self.es_val = ExampleSampler(
            FewRel_data=self.val_data,
            K=self.val_K,
            q=self.val_q
        )
        
        self.entity_marker = EntityMarkerBuilder()
        self.input_feature_builder = InputFeatureBuilder()

    
    def _init_tokenizer(self):
        special_tokens_dict = {
            'marker_start_head': '<subj>',
            'marker_end_head': '</subj>',
            'marker_start_tail': '<obj>',
            'marker_start_tail': '</obj>',
        }
        self.tokenizer.add_special_tokens({"additional_special_tokens": ['<subj>', '</subj>', '<obj>', '</obj>']},)
        special_token_ids_dict = {
            '<subj>': self.tokenizer.convert_tokens_to_ids("<subj>"),
            '</subj>': self.tokenizer.convert_tokens_to_ids("</subj>"),
            '<obj>': self.tokenizer.convert_tokens_to_ids("<obj>"),
            '</obj>': self.tokenizer.convert_tokens_to_ids("</obj>")
        }
        return special_token_ids_dict
    
    def train(
        self,
        model: ProtoFSRE_Entity,
        loss_fn,
        optimizer,
        path_save_model: str
    ):
        encoder = model.get_encoder()
        encoder.resize_token_embeddings(len(self.tokenizer))
        eval_acc_best = [0, 0]
        for t in range(self.epoch):
            self.tprint(f"Epoch: {t + 1} / {self.epoch}, start training!")
            self.train_loop(
                entity_marker=self.entity_marker,
                input_feature_builder=self.input_feature_builder,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            
            self.tprint(f"Start eval!")
            eval_acc = self.eval_loop(
                model=model,
                loss_fn=loss_fn
            )
            if (eval_acc > eval_acc_best[0]) or (eval_acc > eval_acc_best[1]):
                self.tprint("Get A Best Model! Save It!")
                model_filename = f"{t + 1}-{eval_acc:.4f}.pt"
                torch.save(model.state_dict(), path_save_model / model_filename)
                if eval_acc > eval_acc_best[0]:
                    eval_acc_best[0] = eval_acc
                elif eval_acc > eval_acc_best[1]:
                    eval_acc_best[1] = eval_acc

    
    def eval_loop(
        self,
        model: ProtoFSRE_Entity,
        loss_fn,
    ):
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for _ in range(self.val_batch):
                val_input = self._build_episode_input(
                    relation_sampler=self.rs_val,
                    example_sampler=self.es_val,
                    entity_marker=self.entity_marker,
                    input_feature_builder=self.input_feature_builder
                )
                preds = model(
                    support_prompt_inputs=val_input['prompt_input']['support'],
                    query_prompt_inputs=val_input['prompt_input']['query'],
                    support_entity_marker_inputs=val_input['marker_input']['support'],
                    query_entity_marker_inputs=val_input['marker_input']['query'],
                    mask_token_id=self.tokenizer.mask_token_id, 
                    head_start_token_id=self.special_token_ids["<subj>"],
                    head_end_token_id=self.special_token_ids["</subj>"],
                    tail_start_token_id=self.special_token_ids["<obj>"],
                    tail_end_token_id=self.special_token_ids["</obj>"],
                    K=self.val_K
                )
                targets = torch.tensor(
                    self.label_mapping(
                        sampled_relation=val_input['sampled_relation'],
                        targets=val_input['sampled_query_set']['label']
                    )
                )
                loss = loss_fn(preds, targets)
                val_loss += loss
                val_acc += self.get_acc(
                    preds=preds,
                    targets=targets,
                    N=self.val_N,
                    q=self.val_q
                )
            val_loss /= self.val_batch
            val_acc /= self.val_batch
            self.tprint(f"Eval Avg Acc: {val_acc:.5f}, Eval Avg Loss: {val_loss.item():.7f}")
            return val_acc
        
            
    def _build_episode_input(
        self,
        relation_sampler: RelationSampler,
        example_sampler: ExampleSampler,
        entity_marker: EntityMarkerBuilder,
        input_feature_builder: InputFeatureBuilder
    ):
        sampled_relation_list = relation_sampler.sample()
        support_set, query_set = example_sampler.sample(relation=sampled_relation_list)
        #添加rule_file
        rule_file_root = "../../EXER/"
        rule_set = gen_rule_set(rule_file_root)

        prompt_s = PromptBuilder.build_support_prompt_inputs(
                self,
                token_list=support_set['token'],
                head_list=support_set['head'],
                tail_list=support_set['tail'],
                relation_list=support_set['label'],
                mask_token=self.tokenizer.mask_token,
                pid2name=self.pid2name,
                rule_file = rule_set,
        )
        prompt_q = PromptBuilder.build_query_prompt_inputs(
            self,
            token_list=query_set['token'],
            head_list=query_set['head'],
            tail_list=query_set['tail'],
            mask_token=self.tokenizer.mask_token,
            rule_file = rule_set,#change rule_file_URL to a iterable variant
        )
        
        marker_s = entity_marker.mark(
            token_list=support_set['token'],
            head_idx_list=support_set['head_idx'],
            tail_idx_list=support_set['tail_idx'],
        )
        marker_q = entity_marker.mark(
            token_list=query_set['token'],
            head_idx_list=query_set['head_idx'],
            tail_idx_list=query_set['tail_idx'],
        )
        marker_s_text = [' '.join(m) for m in marker_s]
        marker_q_text = [' '.join(m) for m in marker_q]
        
        prompt_input_s = input_feature_builder.build_input_feature(
            tokenizer=self.tokenizer,
            text_list=prompt_s,
        )
        prompt_input_q = input_feature_builder.build_input_feature(
            tokenizer=self.tokenizer,
            text_list=prompt_q,
        )
        marker_input_s = input_feature_builder.build_input_feature(
            tokenizer=self.tokenizer,
            text_list=marker_s_text,
        )
        marker_input_q = input_feature_builder.build_input_feature(
            tokenizer=self.tokenizer,
            text_list=marker_q_text,
        )
        
        return {
            'prompt_input': {
                'support': prompt_input_s, 
                'query': prompt_input_q
            }, 
            'marker_input': {
                'support': marker_input_s, 
                'query': marker_input_q
            },
            'sampled_relation': sampled_relation_list,
            'sampled_support_set': support_set,
            'sampled_query_set': query_set
        }
    
    def train_loop(
        self,
        entity_marker,
        input_feature_builder,
        model,
        loss_fn,
        optimizer,
    ):
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        for t in range(self.batch):
            train_input = self._build_episode_input(
                relation_sampler=self.rs_train,
                example_sampler=self.es_train,
                entity_marker=entity_marker,
                input_feature_builder=input_feature_builder
            )
            with torch.cuda.amp.autocast():
                preds = model(
                    support_prompt_inputs=train_input['prompt_input']['support'],
                    query_prompt_inputs=train_input['prompt_input']['query'],
                    support_entity_marker_inputs=train_input['marker_input']['support'],
                    query_entity_marker_inputs=train_input['marker_input']['query'],
                    mask_token_id=self.tokenizer.mask_token_id, 
                    head_start_token_id=self.special_token_ids["<subj>"],
                    head_end_token_id=self.special_token_ids["</subj>"],
                    tail_start_token_id=self.special_token_ids["<obj>"],
                    tail_end_token_id=self.special_token_ids["</obj>"],
                    K=self.train_K
                )
                targets = torch.tensor(
                    self.label_mapping(
                        sampled_relation=train_input['sampled_relation'],
                        targets=train_input['sampled_query_set']['label']
                    )
                )
                loss = loss_fn(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if t % 10 == 0:
                acc = self.get_acc(
                    preds=preds,
                    targets=targets,
                    N=self.train_N,
                    q=self.train_q
                )
                self.tprint(f"\tBatch:{t} / {self.batch}, Loss: {loss.item():.7f}, Acc: {acc:.5f}")
    
    def label_mapping(self, sampled_relation, targets):
        label_mapping = {c:i for i, c in enumerate(sampled_relation)}
        return [label_mapping[l] for l in targets]
        
    def get_acc(self, preds, targets, N, q):
        correct = (preds.argmax(1) == targets).type(torch.float).sum().item()
        acc = correct / (N * q)
        return acc
    
    def tprint(self, print_str: str):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(time_str + '\t' + print_str)