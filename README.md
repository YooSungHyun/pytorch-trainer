# pytorch-trainer

torch 2.0 and cpu, ddp, deepspeed trainer template

I'm thinking of making a template that is somewhat enforced.
Torch-fabric doesn't support as many features as I thought it would.
Write my own trainer in pure native torch.

Each trainer will be written in its own python file.

>   torch >= 2.1.2</br>
>   cuda 11.8</br>
>   I am experimenting with codebase deepspeed as of 231230.

deepspeed install is,

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After executing the command, **restart the terminal** to install deepspeed.

**Make sure you have 11.8 CUDA and the corresponding TORCH installed,**

```bash
sh scripts/install_deepspeed.sh
```

to install deepspeed.

In `vs_code_launch_json`, upload `launch.json` for debugging vscode.

## Usage

1.   Download raw data and input dir `raw_data`
2.   Copy Model Network(just `nn.Module`) in `networks` dir
3.   make `preprocess.py` and preprocess yourself
4.   if you have to make dataset, check `np_dataset.py` or `pd_dataset.py` and change
5.   if you have to make sampler, check `custom_sampler.py`
     -   i make some useful sampler in `custom_sampler.py` already (reference HF's transformers)
     -   DistributedBucketSampler : make random batch, but lengths same as possible.
     -   LengthGroupedSampler : descending order by `length` column, and select random indices batch. (dynamic batching)
     -   DistributedLengthGroupedSampler: distributed dynamic batching
6.   change `[cpu|ddp|deepspeed]_train.py`
     -   make to dataset, dataloader, etc
     -   **learning rate scheduler must be `{"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": None}`**
     -   `frequency` is step accumulation, if is 2, for every 2 train steps, take 1 scheduler step.
     -   `monitor` is for only `ReduceLROnPlateau`'s loss value
7.   I'm used dict inplace very much. so, use `chk_addr_dict` for compare dict addr. if it is difference, it causes unexpected results. 

# TODO LIST

-   [x] cpu_trainer - lstm example, but it training is weird
-   [x] cpu_trainer - wandb
-   [x] cpu_trainer - continuous learning
-   [x] cpu_trainer - weird lstm training fix ([wandb](https://wandb.ai/bart_tadev/torch-trainer/runs/xrvi8x4g?workspace=user-bart_tadev))
-   [ ] ddp_trainer - lstm or mnist example
-   [ ] ddp_trainer - sampler and dataloader
-   [ ] ddp_trainer = add fp16 and bf16 use
-   [ ] ddp_trainer - training loop additional process?(for distributed learning)
-   [ ] ddp_trainer - wandb have to using gather or something?
-   [ ] ddp_trainer - Reliable wandb logging for distributed learning
-   [ ] deepspeed_trainer - lstm or mnist example
-   [ ] deepspeed_trainer - sampler and dataloader
-   [ ] deepspeed_trainer - training loop additional process?(for distributed learning)
-   [ ] deepspeed_trainer - wandb have to using gather or something?
-   [ ] deepspeed_trainer - Reliable wandb logging for distributed learning
-   [ ] what can i do use accelerator?

# Unsupported list

**tensorboard** - I personally find it too inconvenient.

fsdp - deepspeed is more comfortable for me... ( **I'll consider this if I get more requests**).

# plz help!!!

I don't have much understanding of accelerate, so I'm looking for someone to help me out, PRs are always welcome.

Bugfixes and improvements are always welcome.

**If you can recommend any accelerator related blogs or videos for me to study, I would be grateful. (in issue or someting)**