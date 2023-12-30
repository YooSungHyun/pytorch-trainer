# pytorch-trainer
torch 2.0 and cpu, ddp, deepspeed trainer template

I'm thinking of making a template that is somewhat enforced.
Torch-fabric doesn't support as many features as I thought it would.
Write my own trainer in pure native torch.

Each trainer will be written in its own python file.

>   torch >= 2.1.2
>   cuda 11.8
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

# TODO LIST

-   [x] cpu_trainer - lstm example, but it training is weired
-   [x] cpu_trainer - wandb
-   [x] cpu_trainer - continuous learning
-   [ ] cpu_trainer - weired lstm training fix
-   [ ] ddp_trainer - lstm or mnist example
-   [ ] ddp_trainer - sampler and dataloader
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