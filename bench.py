"""Measure what the training config in train.py actually costs, and what changing it buys.

Three configs, same model, same data, same seed:

  baseline  cpu,  batch 4    exactly what train.py does today
  device    mps,  batch 4    isolates the effect of using the GPU at all
  tuned     mps,  batch 128  isolates the effect of the batch size on top

Writes one JSON per config with per-step loss, per-epoch metrics, and wall-clock,
so the charts in the post come from measurements rather than from a stock image.
"""

import argparse, json, os, time, pathlib
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from model import Net

OUT = pathlib.Path("bench-results")


def loaders(batch_size, workers):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    # Same split and seed for every config, so accuracy differences are the
    # config and not a different validation set.
    train, val = torch.utils.data.random_split(
        full, [45000, 5000], generator=torch.Generator().manual_seed(0)
    )
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers),
        torch.utils.data.DataLoader(val, batch_size=max(batch_size, 256), shuffle=False, num_workers=workers),
    )


def run(name, device, batch_size, epochs, lr, max_steps, workers):
    torch.manual_seed(0)
    dev = torch.device(device)
    net = Net().to(dev)
    opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    trainloader, valloader = loaders(batch_size, workers)

    rec = {
        "name": name, "device": device, "batch_size": batch_size, "lr": lr,
        "epochs": epochs, "steps_per_epoch": len(trainloader),
        "max_steps_per_epoch": max_steps, "epoch_metrics": [], "loss_trace": [],
    }
    print(f"\n=== {name}: {device}, batch {batch_size}, {len(trainloader)} steps/epoch ===")
    t_start = time.perf_counter()

    for epoch in range(epochs):
        net.train()
        running, seen, t0 = 0.0, 0, time.perf_counter()
        for i, (x, y) in enumerate(trainloader):
            if max_steps and i >= max_steps:
                break
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item(); seen += 1
            # Sample the trace rather than storing every step; enough points to
            # draw a curve, few enough to keep the JSON readable.
            if i % max(1, len(trainloader) // 200) == 0:
                rec["loss_trace"].append({"epoch": epoch, "step": i, "loss": round(loss.item(), 4)})
        if device == "mps":
            torch.mps.synchronize()
        train_s = time.perf_counter() - t0
        imgs = seen * batch_size

        net.eval()
        vloss, correct, total = 0.0, 0, 0
        t1 = time.perf_counter()
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(dev), y.to(dev)
                out = net(x)
                vloss += crit(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        if device == "mps":
            torch.mps.synchronize()
        val_s = time.perf_counter() - t1

        m = {
            "epoch": epoch + 1,
            "train_loss": round(running / max(seen, 1), 4),
            "val_loss": round(vloss / len(valloader), 4),
            "val_acc": round(100 * correct / total, 2),
            "train_seconds": round(train_s, 2),
            "val_seconds": round(val_s, 2),
            "images_per_second": round(imgs / train_s, 1),
            "steps_run": seen,
        }
        rec["epoch_metrics"].append(m)
        print(f"  epoch {m['epoch']}  loss {m['train_loss']:.4f}  val {m['val_loss']:.4f}  "
              f"acc {m['val_acc']:.2f}%  {m['train_seconds']:.1f}s  {m['images_per_second']:.0f} img/s")

    rec["total_seconds"] = round(time.perf_counter() - t_start, 2)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"  total {rec['total_seconds']:.1f}s -> bench-results/{name}.json")
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=["baseline", "device", "tuned"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch (0 = full)")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    cfg = {
        "baseline": ("cpu", 4, 0.001),
        "device":   ("mps", 4, 0.001),
        "tuned":    ("mps", 128, 0.01),   # lr scaled with batch size
    }[a.config]
    run(a.config, cfg[0], cfg[1], a.epochs, cfg[2], a.max_steps, a.workers)
