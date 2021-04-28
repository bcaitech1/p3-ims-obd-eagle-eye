import torch, gc


def gpu():
    try:
        del trainer
    except:
        pass

    try:
        del model
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # termianl 명령어
    # nvidia-smi
    # 10초마다 확인
    # watch -n 10 nvidia-smi
    # watch -n 0.5 -d nvidia-smi


if __name__ == "__main__":
    gpu()
