import os
import torch
import torch.distributed as dist
import time

def main():
    # OMPI環境変数からRANKとWORLD_SIZEを取得し、torch.distributed用に設定
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # MASTER_ADDRとMASTER_PORTを設定（必要に応じて環境変数から取得）
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "8888")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # プロセスグループの初期化
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"Rank: {dist.get_rank()}, Size: {dist.get_world_size()}")

    # 使用するGPUの設定
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("CUDA対応GPUが見つかりません。")
    device = rank % ngpus
    torch.cuda.set_device(device)

    # テンソルのサイズを確認（最初は小さなサイズでテスト）
    tensor_size = 10 * 1024 * 1024 * 1024 // 4  # 10GBのfloat32テンソル
    try:
        x = torch.randn(tensor_size, device=device)
        print(f'rank {rank}: Tensor of size {x.size()} created on device {device}')
    except RuntimeError as e:
        print(f'rank {rank}: テンソルの作成に失敗しました: {e}')
        dist.destroy_process_group()
        return
    x = x.to(torch.int8)

    # ブロードキャストの前後で通信時間を計測
    torch.cuda.synchronize(device)
    start_time = time.time()

    dist.broadcast(x, src=0)

    torch.cuda.synchronize(device)
    end_time = time.time()

    # 通信時間の出力
    comm_time = end_time - start_time
    print(f'rank {rank}: Broadcast communication time: {comm_time:.4f} seconds')

    # プロセスグループの終了
    dist.destroy_process_group()

if __name__ == "__main__":
    main()