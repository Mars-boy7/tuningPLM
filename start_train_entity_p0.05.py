import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train import ProtoEntityTrainer
from pathlib import Path
from transformers import AutoTokenizer
from model.proto import ProtoFSRE_Entity
import torch
import time
from utils import set_device, setup_seed, load_json_data


def main():
    setup_seed(20)
    set_device()
    
    timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
    path_project = Path("/home/zjc/test_tuningPLM")
    path_dict = {
        'train_data': path_project / "data" / "FewRel 1.0" / "sampled_data0.05.json",
        'val_data': path_project / "data" / "FewRel 1.0" / "sampled_data_val0.05.json",
        "pid2name": path_project / "data" / "pid2name.json",
        "experiment_result": path_project / "experiment_result" / timestamp,
    }
    os.makedirs(path_dict["experiment_result"], exist_ok=True)
    
    train_data = load_json_data(path_dict['train_data'])
    val_data = load_json_data(path_dict["val_data"])
    pid2name = load_json_data(path_dict["pid2name"])
    
    encoder_name = "/home/zjc/test_tuningPLM/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = ProtoFSRE_Entity(encoder_name=encoder_name, dropout=0.05)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    trainer = ProtoEntityTrainer(
        train_data=train_data,
        val_data=val_data,
        pid2name=pid2name,
        tokenizer=tokenizer,
        train_N=10,
        train_K=1,
        train_q=5,
        epoch=1,
        batch=100,
        val_N=5,
        val_K=1,
        val_q=1,
        val_batch=500,
    )
    trainer.train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        path_save_model=path_dict["experiment_result"]
    )


if __name__ == '__main__':
    main()