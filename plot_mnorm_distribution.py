import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from optimizers.lion import Lion  # 仮定: Lionクラスはこのコンテキストで使用されていない
from model import GPTConfig, GPT
import seaborn as sns
import numpy as np

# sns.set()
# sns.set_context("paper", font_scale=0.75, rc={"lines.linewidth": 4})

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "axes.labelsize": 6,     # reduce label size
    "font.size": 6,          # reduce general font size
    "legend.fontsize": 6,    # reduce legend font size
})

def exists(val):
    return val is not None

def main():
    folder = './out_small_lion_100k/2024-02-09-14:47:06.606385'
    epoch = 10000
    file_name = f'{folder}/ckpt_{epoch}.pt'

    ckpt = torch.load(file_name, map_location=torch.device('cpu'))

    checkpoint_model_args = ckpt['model_args']
    config_dic = ckpt['config']
    model_state_dict = ckpt['model']
    optimizer_state_dict = ckpt['optimizer']
    device_type = 'cpu'
    # model init
    model_args = {}
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(model_state_dict.items()):
        if k.startswith(unwanted_prefix):
            model_state_dict[k[len(unwanted_prefix):]] = model_state_dict.pop(k)
    model.load_state_dict(model_state_dict)

    optimizer = model.configure_optimizers(config_dic['optimizer_name'], 0.1, 0.1, (0.9, 0.99), 0, device_type)
    optimizer.load_state_dict(optimizer_state_dict)

    param_name_dict = {}
    for name, param in model.named_parameters():
        param_name_dict[param] = name
    
    exp_avg_l1_norms = []
    exp_avg_l1_norms_max = []
    exp_distribution = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if "exp_avg" in state:
                exp_avg = optimizer.state[p]["exp_avg"]
                exp_avg_norm_l1 = torch.abs(exp_avg).mean(dtype=torch.float32).item()
                exp_avg_l1_norms.append(exp_avg_norm_l1)

                exp_avg_norm_l1_max = torch.abs(exp_avg).max().item()
                exp_avg_l1_norms_max.append(exp_avg_norm_l1_max)

                lname = param_name_dict[p]
                if lname in ['transformer.wpe.weight', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.ln_1.weight']:
                    exp_distribution[lname] = exp_avg

    fig, axs = plt.subplots(1, 4, figsize=(6, 2))

    # exp_avgのL1ノルムの分布、i=0は通常スケール、i=1はログスケール
    axs[0].hist(exp_avg_l1_norms, bins=30, alpha=0.75)
    axs[0].set_title('Mean exp_avg L1 Norms')
    axs[0].set_xlabel('Norm Value')

    exp_avg_l1_norms_log = np.log10(exp_avg_l1_norms)
    axs[1].hist(exp_avg_l1_norms_log, bins=30, alpha=0.75)  # log=Trueを追加
    axs[1].set_title('Mean exp_avg L1 Norms\n(Log Scale)')
    axs[1].set_xlabel(f'log_10 (Norm Value)')

    # exp_avgのL1ノルムの分布、i=0は通常スケール、i=1はログスケール
    axs[2].hist(exp_avg_l1_norms_max, bins=30, alpha=0.75)
    axs[2].set_title('Max exp_avg L1 Norms')
    axs[2].set_xlabel('Norm Value')

    exp_avg_l1_norms_max_log = np.log10(exp_avg_l1_norms_max)
    axs[3].hist(exp_avg_l1_norms_max_log, bins=30, alpha=0.75)  # log=Trueを追加
    axs[3].set_title('Max exp_avg L1 Norms\n(Log Scale)')
    axs[3].set_xlabel(f'log_10 (Norm Value)')

    plt.tight_layout()

    # PDFとして保存
    with PdfPages('./graphs/norms_distribution.pdf') as pdf:
        pdf.savefig()
        plt.close()

    fig, axs = plt.subplots(1, len(exp_distribution), figsize=(6, 2))
    idx = 0
    for (name, pdis) in exp_distribution.items():
        # exp_avgのL1ノルムの分布、i=0は通常スケール、i=1はログスケール
        axs[idx].hist(pdis)
        axs[idx].set_title(f'{name.replace("transformer.","").replace("weight","")}')
        axs[idx].set_xlabel('Norm Value')
        idx += 1
    with PdfPages('./graphs/mom_distribution.pdf') as pdf:
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    main()
