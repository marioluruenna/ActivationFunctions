from __future__ import annotations
import time
import json
import shutil
import random
import math
import re
import hashlib
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import copy
import sys

# =========================
# CONFIGURACIÓN (EDITA AQUÍ)
# =========================

# --- Logging a fichero ---
LOG_TO_FILE = True
LOG_FILENAME = "log_twish.txt"
LOG_APPEND = True  # True => añade al final, False => sobrescribe

# --- Dataset ---
IMAGENET_ROOT = "/home/mario/TFM/datasets/imagenet"

SUBSET_MODE = "first_k"      # "first_k" | "explicit_list"
SUBSET_K = 100
SUBSET_SYNSETS = [
    # "n01440764",
]

SUBSET_NAME = f"_subset_{SUBSET_MODE}_{SUBSET_K if SUBSET_MODE=='first_k' else 'custom'}"
RECREATE_SUBSET_DIR = True

# --- Modelo ---
ARCH = "resnet50"
ACTIVATION = "twish"          # "relu" | "swish" | "twish"
ZERO_INIT_RESIDUAL = False

# --- Entrenamiento ---
EPOCHS = 80 # 50 epochs para 10 clases, 80 para 100 clases y 100 para 1000 clases

BATCH_SIZE = 64 #128 para relu, 64 para el resto

# Grad accumulation para batch efectivo grande
GRAD_ACCUM_STEPS = 4 #2 para relu, 4 para el resto

NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# --- Optimización ---
OPTIMIZER = "sgd"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

BASE_LR_256 = 0.1
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS
LR = BASE_LR_256 * (EFFECTIVE_BATCH / 256.0)

# --- Scheduler (RECIPE: step por iteraciones/updates, no por epoch) ---
SCHEDULER = "iter_step"

# Decays del recipe expresados en PORCENTAJE del entrenamiento
LR_DECAY_FRACTIONS = [0.25, 0.65, 0.85, 0.95] # [0.50, 0.75, 0.90, 0.95] para 10 clases, [0.35, 0.65, 0.85, 0.95] para 100 clases y [0.30, 0.60, 0.80, 0.90] para 1000 clases
LR_DECAY_GAMMA = 0.2 # 0.2 para 10 clases, 0.15 para 100 clases y 0.1 para 1000 clases

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

LABEL_SMOOTHING = 0.10  # 0.05 para 10 clases, 0.10 para 100 y 1000 clases

USE_RANDAUGMENT = True # True para 10 y 100 clases, False para 1000
RA_NUM_OPS = 2 # 2 para 10 y 100 clases, se desactiva para 1000 clases (Es de RandAugment)
RA_MAGNITUDE = 9 # 5 para 10 clases, 9 para 100 clases y se desactiva para 1000 clases (Es de RandAugment)

USE_AUTOAUGMENT = False  # False para 10 y 100 clases, True para 1000

EMA_DECAY = 0.999  # 0.0 para 10 clases, 0.999 para 100 clases y 0.9999 para 1000 clases

# --- AMP ---
AMP_ENABLED = True

# --- Reproducibilidad ---
SEED = 123
DETERMINISTIC = False

# --- Output / checkpoints ---
OUT_DIR = "/home/mario/TFM/results/resnet_50_twish_100_classes"
SAVE_EVERY_EPOCH = True
PRINT_FREQ = 50

# --- Logs de estado / preflight ---
VERBOSE_STATUS = True
PREFETCH_FIRST_BATCH = True
PREFETCH_TIMEOUT_SECS = 300

# --- Auto-resume / Ctrl+C ---
AUTO_RESUME = True
AUTO_RESUME_STRICT = True
EXIT_IF_ALREADY_FINISHED = True

SAVE_INTERRUPT_CKPT = True
INTERRUPT_CKPT_NAME = "ckpt_interrupt.pt"
PREFER_INTERRUPT_CKPT = True


# =========================
# LOGGING DE ESTADO
# =========================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def status(msg: str) -> None:
    if VERBOSE_STATUS:
        print(f"[{_ts()}] {msg}", flush=True)


class Stage:
    """Context manager para imprimir 'Cargando X...' y 'X cargado (t=...)'."""
    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self):
        status(f"{self.name}...")
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - (self.t0 or time.time())
        if exc_type is None:
            status(f"{self.name} ✓  (t={dt:.2f}s)")
        else:
            status(f"{self.name} ✗  (t={dt:.2f}s) -> ERROR: {exc}")
        return False


# =========================
# AUTO-RESUME HELPERS
# =========================

def make_run_id(cfg: dict) -> str:
    """
    ID estable del experimento para evitar reanudar el experimento equivocado.
    Importante: si cambias subset/model/etc., run_id cambia => resume estricto fallará (bien).
    """
    keys_for_id = {
        "IMAGENET_ROOT": cfg.get("IMAGENET_ROOT"),
        "SUBSET_MODE": cfg.get("SUBSET_MODE"),
        "SUBSET_K": cfg.get("SUBSET_K"),
        "SUBSET_SYNSETS": cfg.get("SUBSET_SYNSETS"),
        "SUBSET_NAME": cfg.get("SUBSET_NAME"),
        "ARCH": cfg.get("ARCH"),
        "ACTIVATION": cfg.get("ACTIVATION"),
        "ZERO_INIT_RESIDUAL": cfg.get("ZERO_INIT_RESIDUAL"),
        "EPOCHS": cfg.get("EPOCHS"),
        "BATCH_SIZE": cfg.get("BATCH_SIZE"),
        "GRAD_ACCUM_STEPS": cfg.get("GRAD_ACCUM_STEPS"),
        "EFFECTIVE_BATCH": cfg.get("EFFECTIVE_BATCH"),
        "OPTIMIZER": cfg.get("OPTIMIZER"),
        "LR": cfg.get("LR"),
        "MOMENTUM": cfg.get("MOMENTUM"),
        "WEIGHT_DECAY": cfg.get("WEIGHT_DECAY"),
        "SCHEDULER": cfg.get("SCHEDULER"),
        "LR_DECAY_FRACTIONS": cfg.get("LR_DECAY_FRACTIONS"),
        "LR_DECAY_UPDATES": cfg.get("LR_DECAY_UPDATES"),
        "LR_DECAY_GAMMA": cfg.get("LR_DECAY_GAMMA"),
        "MAX_UPDATES": cfg.get("MAX_UPDATES"),
        "UPDATES_PER_EPOCH": cfg.get("UPDATES_PER_EPOCH"),
        "NORM_MEAN": cfg.get("NORM_MEAN"),
        "NORM_STD": cfg.get("NORM_STD"),
        "LABEL_SMOOTHING": cfg.get("LABEL_SMOOTHING"),
        "USE_RANDAUGMENT": cfg.get("USE_RANDAUGMENT"),
        "RA_NUM_OPS": cfg.get("RA_NUM_OPS"),
        "RA_MAGNITUDE": cfg.get("RA_MAGNITUDE"),
        "USE_AUTOAUGMENT": cfg.get("USE_AUTOAUGMENT"),
        "EMA_DECAY": cfg.get("EMA_DECAY"),
    }
    s = json.dumps(keys_for_id, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def find_latest_epoch_ckpt(out_dir: Path) -> Tuple[Optional[Path], int]:
    """
    Devuelve (path, epoch) del ckpt_epoch_XXX.pt con XXX mayor.
    """
    best_path = None
    best_ep = -1
    if not out_dir.exists():
        return None, -1

    for p in out_dir.glob("ckpt_epoch_*.pt"):
        m = re.search(r"ckpt_epoch_(\d+)\.pt$", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best_path = p

    return best_path, best_ep


def move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def load_checkpoint_for_resume(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler,
    scaler,
    ema,
    device: torch.device,
    expected_run_id: str,
    strict: bool,
    current_classes: Optional[List[str]] = None,
) -> Tuple[int, float, int]:
    """
    Carga checkpoint y devuelve:
      start_epoch = ckpt_epoch + 1
      best_global_acc
      global_update_step
    """
    status(f"Auto-resume: cargando checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    ckpt_cfg = ckpt.get("config", {})
    ckpt_run_id = ckpt_cfg.get("run_id", None)

    if strict:
        if ckpt_run_id is None:
            raise RuntimeError("Checkpoint no tiene config.run_id. No puedo hacer resume estricto.")
        if ckpt_run_id != expected_run_id:
            raise RuntimeError(
                f"Checkpoint run_id={ckpt_run_id} != run_id actual={expected_run_id}. "
                f"(Evita reanudar un experimento distinto.)"
            )
    else:
        if ckpt_run_id is not None and ckpt_run_id != expected_run_id:
            status(f"AVISO: run_id ckpt ({ckpt_run_id}) != actual ({expected_run_id}) pero strict=False => continúo.")

    # Modelo
    model.load_state_dict(ckpt["model"], strict=True)

    # Optimizer
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        move_optimizer_state_to_device(optimizer, device)

    # Scheduler
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    # AMP scaler
    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            status(f"AVISO: no pude cargar scaler state ({e}). Continúo con scaler nuevo.")

    # EMA
    if ema is not None and ckpt.get("ema") is not None:
        try:
            ema.load_state_dict(ckpt["ema"])
        except Exception as e:
            status(f"AVISO: no pude cargar EMA state ({e}). Continúo con EMA reiniciada.")

    # Best metric
    best = float(ckpt.get("best_global_accuracy", -1.0))

    last_epoch = int(ckpt.get("epoch", 0))
    start_epoch = last_epoch + 1

    # Update step (iteraciones/updates)
    global_update_step = int(ckpt.get("global_update_step", 0))

    # Check clases
    ckpt_classes = ckpt.get("classes", None)
    if current_classes is not None and ckpt_classes is not None:
        if list(ckpt_classes) != list(current_classes):
            status("AVISO SERIO: las clases del checkpoint no coinciden con las actuales.")
            status("Esto puede invalidar el entrenamiento (fc/labels distintas).")

    status(
        f"Auto-resume ✓ last_epoch={last_epoch} -> start_epoch={start_epoch} | "
        f"best_global_acc={best:.2f}% | global_update_step={global_update_step}"
    )
    return start_epoch, best, global_update_step


def pick_resume_checkpoint(
    out_dir: Path,
    expected_run_id: str,
    strict: bool,
) -> Tuple[Optional[Path], Optional[int], str]:
    """
    Decide qué checkpoint usar para reanudar.
    Preferencia:
      1) ckpt_interrupt.pt (si existe y pasa validación run_id)
      2) ckpt_epoch_XXX.pt con XXX mayor (si existe)
    Devuelve (path, epoch_in_ckpt, reason_msg)
    """
    interrupt_path = out_dir / INTERRUPT_CKPT_NAME
    latest_epoch_path, latest_ep = find_latest_epoch_ckpt(out_dir)

    # 1) Intentar interrupt primero
    if PREFER_INTERRUPT_CKPT and interrupt_path.exists():
        try:
            ckpt = torch.load(str(interrupt_path), map_location="cpu")
            ckpt_cfg = ckpt.get("config", {})
            ckpt_run_id = ckpt_cfg.get("run_id", None)
            ckpt_epoch = int(ckpt.get("epoch", 0))

            if strict:
                if ckpt_run_id is None:
                    raise RuntimeError("ckpt_interrupt.pt no tiene run_id")
                if ckpt_run_id != expected_run_id:
                    raise RuntimeError(f"ckpt_interrupt.pt run_id={ckpt_run_id} != actual={expected_run_id}")

            return interrupt_path, ckpt_epoch, f"Detectado {INTERRUPT_CKPT_NAME} (epoch={ckpt_epoch})"

        except Exception as e:
            status(f"AVISO: existe {INTERRUPT_CKPT_NAME} pero no es usable para resume ({e}). Ignorándolo.")

    # 2) Si no, usar último epoch
    if latest_epoch_path is not None and latest_ep >= 1:
        return latest_epoch_path, latest_ep, f"Detectado {latest_epoch_path.name} (epoch={latest_ep})"

    return None, None, "No se han detectado checkpoints"


# =========================
# AMP HELPERS
# =========================

class _NullScaler:
    """GradScaler no-op para CPU o AMP desactivado."""
    def scale(self, x): return x
    def unscale_(self, optimizer): return None
    def step(self, optimizer): optimizer.step()
    def update(self): return None
    def state_dict(self): return {}
    def load_state_dict(self, state): return None


def make_grad_scaler(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return _NullScaler()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


@contextmanager
def autocast_ctx(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        yield
        return
    try:
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            yield
    except TypeError:
        with torch.cuda.amp.autocast(enabled=True):
            yield


# =========================
# MÉTRICAS TOP-K (micro, exactas)
# =========================

def topk_correct_counts(outputs: torch.Tensor, targets: torch.Tensor, ks: List[int]) -> Dict[int, int]:
    ks = sorted(set(int(k) for k in ks))
    maxk = max(ks)
    pred = outputs.topk(maxk, dim=1, largest=True, sorted=True).indices  # [B,maxk]
    correct = pred.eq(targets.view(-1, 1))  # [B,maxk]
    out: Dict[int, int] = {}
    for k in ks:
        out[k] = correct[:, :k].any(dim=1).sum().item()
    return out


def topk_list_for_num_classes(num_classes: int) -> Tuple[int, int, int]:
    k1 = 1
    k5 = min(5, num_classes)
    k10 = min(10, num_classes)
    return k1, k5, k10


def fmt_metric_labels(k1: int, k5: int, k10: int) -> Tuple[str, str, str]:
    return f"top{k1}", f"top{k5}", f"top{k10}"


# =========================
# RESNET50 MANUAL
# =========================
class twish(nn.Module):
    def __init__(self, beta=0.25, gamma=0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        return x * torch.tanh(self.beta * x) + self.gamma * x

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("swish", "silu"):
        return nn.SiLU(inplace=True)
    if name == ("twish"):
        return twish(beta=1.0, gamma=1.0)
    raise ValueError(f"Activación no soportada: {name}")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.act = get_activation(activation)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        outplanes = planes * self.expansion
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        activation: str = "relu",
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.activation_name = activation
        self.act = get_activation(activation)

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes=64,  blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        outplanes = planes * block.expansion

        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers_list = []
        layers_list.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                activation=self.activation_name,
            )
        )
        self.inplanes = outplanes

        for _ in range(1, blocks):
            layers_list.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    activation=self.activation_name,
                )
            )

        return nn.Sequential(*layers_list)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50_custom(num_classes: int, activation: str = "relu", zero_init_residual: bool = False) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        activation=activation,
        zero_init_residual=zero_init_residual,
    )


# =========================
# SCHEDULER (ITERATIONS / UPDATES)
# =========================

class IterStepLRScheduler:
    def __init__(self, base_lr: float, milestones: List[int], gamma: float, max_updates: int):
        self.base_lr = float(base_lr)
        self.milestones = sorted(int(x) for x in milestones)
        self.gamma = float(gamma)
        self.max_updates = int(max_updates)
        self.global_update_step = 0

    def lr_for_update(self, u: int) -> float:
        n = bisect_right(self.milestones, int(u))
        return self.base_lr * (self.gamma ** n)

    def set_lr_for_current_update(self, optimizer: optim.Optimizer) -> float:
        lr = self.lr_for_update(self.global_update_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def mark_update_done(self) -> None:
        self.global_update_step += 1

    def step(self) -> None:
        return

    def state_dict(self) -> dict:
        return {
            "base_lr": self.base_lr,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "max_updates": self.max_updates,
            "global_update_step": self.global_update_step,
        }

    def load_state_dict(self, d: dict) -> None:
        self.base_lr = float(d.get("base_lr", self.base_lr))
        self.milestones = sorted(int(x) for x in d.get("milestones", self.milestones))
        self.gamma = float(d.get("gamma", self.gamma))
        self.max_updates = int(d.get("max_updates", self.max_updates))
        self.global_update_step = int(d.get("global_update_step", 0))


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            src = msd[k]
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(src.detach(), alpha=(1.0 - self.decay))
            else:
                v.copy_(src)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "ema": self.ema.state_dict()}

    def load_state_dict(self, d: dict):
        self.decay = float(d.get("decay", self.decay))
        self.ema.load_state_dict(d["ema"], strict=True)


# =========================
# UTILIDADES
# =========================

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def init_device_with_status() -> torch.device:
    if not torch.cuda.is_available():
        status("GPU no disponible: torch.cuda.is_available()=False. Usando CPU.")
        return torch.device("cpu")

    status("GPU disponible: torch.cuda.is_available()=True")
    status("Iniciando GPU...")

    try:
        _ = torch.cuda.current_device()
        _ = torch.empty(1, device="cuda")
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        status(f"GPU iniciada ✓  -> {name} | VRAM={vram_gb:.2f} GB")
    except Exception as e:
        status(f"GPU disponible pero falló la inicialización: {e}. Usando CPU.")
        return torch.device("cpu")

    return torch.device("cuda")


def imagenet_transforms(train: bool = True) -> transforms.Compose:

    mean = NORM_MEAN
    std  = NORM_STD

    if train:
        augs = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]

        if USE_AUTOAUGMENT:
            augs.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))

        if USE_RANDAUGMENT:
            augs.append(transforms.RandAugment(num_ops=RA_NUM_OPS, magnitude=RA_MAGNITUDE))

        augs += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        return transforms.Compose(augs)

    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def list_synsets(train_dir: Path) -> List[str]:
    synsets = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not synsets:
        raise RuntimeError(f"No encuentro carpetas de clase en: {train_dir}")
    return synsets


def build_subset_symlinks(
    imagenet_root: Path,
    subset_name: str,
    selected_synsets: List[str],
    recreate: bool = True,
) -> Tuple[Path, Path]:
    train_src = imagenet_root / "train"
    val_src = imagenet_root / "val"

    subset_root = imagenet_root / subset_name
    subset_train = subset_root / "train"
    subset_val = subset_root / "val"

    if recreate and subset_root.exists():
        shutil.rmtree(subset_root)

    subset_train.mkdir(parents=True, exist_ok=True)
    subset_val.mkdir(parents=True, exist_ok=True)

    for s in selected_synsets:
        src_tr = train_src / s
        src_va = val_src / s
        if not src_tr.exists():
            raise RuntimeError(f"Falta clase en train: {src_tr}")
        if not src_va.exists():
            raise RuntimeError(f"Falta clase en val: {src_va}")

        dst_tr = subset_train / s
        dst_va = subset_val / s

        if not dst_tr.exists():
            dst_tr.symlink_to(src_tr, target_is_directory=True)
        if not dst_va.exists():
            dst_va.symlink_to(src_va, target_is_directory=True)

    return subset_train, subset_val


def make_dataloaders(train_dir: Path, val_dir: Path, device: torch.device) -> Tuple[DataLoader, DataLoader, List[str]]:
    ds_train = ImageFolder(str(train_dir), transform=imagenet_transforms(train=True))
    ds_val = ImageFolder(str(val_dir), transform=imagenet_transforms(train=False))

    if ds_train.classes != ds_val.classes:
        raise RuntimeError("Las clases de train y val no coinciden en el subset (revisa symlinks/estructura).")

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and device.type == "cuda"),
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and device.type == "cuda"),
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        drop_last=False,
    )
    return dl_train, dl_val, ds_train.classes


def prefetch_one_batch(loader: DataLoader, device: torch.device, name: str) -> None:
    with Stage(f"Precargando 1er batch ({name})"):
        t0 = time.time()
        it = iter(loader)
        while True:
            try:
                images, targets = next(it)
                break
            except StopIteration:
                raise RuntimeError(f"{name}: loader vacío.")
            except Exception as e:
                if time.time() - t0 > PREFETCH_TIMEOUT_SECS:
                    raise RuntimeError(f"{name}: timeout precargando batch: {e}") from e

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _ = images.mean() + targets.float().mean()

        if device.type == "cuda":
            torch.cuda.synchronize()


def build_model(num_classes: int) -> nn.Module:
    if ARCH.lower() == "resnet50":
        return resnet50_custom(
            num_classes=num_classes,
            activation=ACTIVATION,
            zero_init_residual=ZERO_INIT_RESIDUAL,
        )
    raise ValueError(f"Arquitectura no soportada: {ARCH}")


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    if OPTIMIZER.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Optimizador no soportado: {OPTIMIZER}")


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


# =========================
# LOOP DE ENTRENAMIENTO
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: IterStepLRScheduler,
    device: torch.device,
    scaler,
    epoch: int,
    num_classes: int,
    amp_enabled: bool,
    ema,
) -> Tuple[float, float, float, float, int]:

    model.train()

    total = 0
    loss_sum = 0.0

    k1, k5, k10 = topk_list_for_num_classes(num_classes)
    ks = [k1, k5, k10]
    s_top1, s_top5, s_top10 = fmt_metric_labels(k1, k5, k10)

    correct_counts = {k: 0 for k in sorted(set(ks))}

    optimizer.zero_grad(set_to_none=True)

    num_batches = len(loader)
    rem = num_batches % GRAD_ACCUM_STEPS

    def denom_for_batch_index(i: int) -> int:
        if rem == 0:
            return GRAD_ACCUM_STEPS
        if i >= num_batches - rem:
            return rem
        return GRAD_ACCUM_STEPS

    for i, (images, targets) in enumerate(loader):
        if scheduler.global_update_step >= scheduler.max_updates:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        denom = float(denom_for_batch_index(i))

        with autocast_ctx(amp_enabled, device):
            outputs = model(images)
            loss_raw = criterion(outputs, targets)
            loss = loss_raw / denom

        scaler.scale(loss).backward()

        do_step = ((i + 1) % GRAD_ACCUM_STEPS == 0) or (i == num_batches - 1)
        if do_step:
            _ = scheduler.set_lr_for_current_update(optimizer)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            scheduler.mark_update_done()

        bs = targets.size(0)
        total += bs
        loss_sum += float(loss_raw.item()) * bs

        batch_counts = topk_correct_counts(outputs, targets, ks=sorted(set(ks)))
        for k, c in batch_counts.items():
            correct_counts[k] += int(c)

        if (i + 1) % PRINT_FREQ == 0:
            avg_loss = loss_sum / max(1, total)
            avg_t1 = 100.0 * correct_counts[k1] / max(1, total)
            avg_t5 = 100.0 * correct_counts[k5] / max(1, total)
            avg_t10 = 100.0 * correct_counts[k10] / max(1, total)
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"[train][e{epoch:03d}][{i+1:05d}/{len(loader):05d}] "
                f"loss={avg_loss:.4f} | "
                f"{s_top1}={avg_t1:.2f}% | "
                f"{s_top5}={avg_t5:.2f}% | "
                f"{s_top10}={avg_t10:.2f}% | "
                f"lr={lr_now:.6f} | "
                f"updates={scheduler.global_update_step}/{scheduler.max_updates}",
                flush=True
            )

    avg_loss = loss_sum / max(1, total)
    top1 = 100.0 * correct_counts[k1] / max(1, total)
    top5 = 100.0 * correct_counts[k5] / max(1, total)
    top10 = 100.0 * correct_counts[k10] / max(1, total)

    return avg_loss, top1, top5, top10, int(scheduler.global_update_step)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_classes: int,
    amp_enabled: bool,
) -> Tuple[float, float, float, float, float]:

    model.eval()

    total = 0
    loss_sum = 0.0

    k1, k5, k10 = topk_list_for_num_classes(num_classes)
    ks = [k1, k5, k10]
    s_top1, s_top5, s_top10 = fmt_metric_labels(k1, k5, k10)

    correct_counts = {k: 0 for k in sorted(set(ks))}

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_ctx(amp_enabled, device):
            outputs = model(images)
            loss = criterion(outputs, targets)

        bs = targets.size(0)
        total += bs
        loss_sum += float(loss.item()) * bs

        batch_counts = topk_correct_counts(outputs, targets, ks=sorted(set(ks)))
        for k, c in batch_counts.items():
            correct_counts[k] += int(c)

    avg_loss = loss_sum / max(1, total)
    top1 = 100.0 * correct_counts[k1] / max(1, total)
    top5 = 100.0 * correct_counts[k5] / max(1, total)
    top10 = 100.0 * correct_counts[k10] / max(1, total)

    global_accuracy = top1

    print(
        f"[val][e{epoch:03d}] "
        f"loss={avg_loss:.4f} | "
        f"{s_top1}={top1:.2f}% | "
        f"{s_top5}={top5:.2f}% | "
        f"{s_top10}={top10:.2f}% | "
        f"global_accuracy (val, aciertos/total)={global_accuracy:.2f}%",
        flush=True
    )

    return avg_loss, top1, top5, top10, global_accuracy

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# =========================
# MAIN
# =========================

def main():
    with Stage("Inicializando semilla y flags"):
        set_seed(SEED, deterministic=DETERMINISTIC)

    with Stage("Detectando e inicializando dispositivo"):
        device = init_device_with_status()
        amp_enabled = bool(AMP_ENABLED and device.type == "cuda")
        status(f"Dispositivo seleccionado: {device} | AMP={amp_enabled} | activation={ACTIVATION}")

    imagenet_root = Path(IMAGENET_ROOT).expanduser()
    train_dir = imagenet_root / "train"
    val_dir = imagenet_root / "val"

    with Stage("Comprobando rutas del dataset"):
        if not imagenet_root.exists():
            raise RuntimeError(f"IMAGENET_ROOT no existe: {IMAGENET_ROOT}")
        if not train_dir.is_dir():
            raise RuntimeError(f"No encuentro train/: {train_dir}")
        if not val_dir.is_dir():
            raise RuntimeError(f"No encuentro val/: {val_dir}")
        status(f"IMAGENET_ROOT OK: {imagenet_root}")

    with Stage("Listando clases disponibles (synsets)"):
        all_synsets = list_synsets(train_dir)
        status(f"Total clases disponibles: {len(all_synsets)}")

    with Stage("Seleccionando subset de clases"):
        if SUBSET_MODE == "first_k":
            selected = all_synsets[:SUBSET_K]
        elif SUBSET_MODE == "explicit_list":
            if not SUBSET_SYNSETS:
                raise RuntimeError("SUBSET_MODE='explicit_list' pero SUBSET_SYNSETS está vacío.")
            selected = SUBSET_SYNSETS
        else:
            raise ValueError(f"SUBSET_MODE no válido: {SUBSET_MODE}")
        status(f"Clases seleccionadas ({len(selected)}): {selected}")

    with Stage("Creando subset (symlinks) train/val"):
        subset_train_dir, subset_val_dir = build_subset_symlinks(
            imagenet_root=imagenet_root,
            subset_name=SUBSET_NAME,
            selected_synsets=selected,
            recreate=RECREATE_SUBSET_DIR,
        )
        status(f"Subset train: {subset_train_dir}")
        status(f"Subset val:   {subset_val_dir}")

    with Stage("Cargando datasets y dataloaders"):
        dl_train, dl_val, classes = make_dataloaders(subset_train_dir, subset_val_dir, device=device)
        num_classes = len(classes)
        status(f"Num classes (subset): {num_classes}")
        status(f"Train batches/epoch: {len(dl_train)} | Val batches: {len(dl_val)}")
        status(f"Batch físico={BATCH_SIZE} | GradAccum={GRAD_ACCUM_STEPS} | Batch efectivo={EFFECTIVE_BATCH}")
        status(f"LR(base para batch efectivo 256)={BASE_LR_256} | LR actual (escalado)={LR:.6f}")
        status(f"LR decay fractions (recipe): {LR_DECAY_FRACTIONS} | gamma={LR_DECAY_GAMMA}")

    # -------------------------
    # CALCULO DE UPDATES TOTALES + MILESTONES POR PORCENTAJE
    # -------------------------
    # updates_per_epoch = ceil(num_batches / GRAD_ACCUM_STEPS)
    updates_per_epoch = int(math.ceil(len(dl_train) / float(GRAD_ACCUM_STEPS)))
    total_updates = int(EPOCHS * updates_per_epoch)

    lr_decay_updates = [max(1, int(round(f * total_updates))) for f in LR_DECAY_FRACTIONS]
    lr_decay_updates = sorted(set(lr_decay_updates))

    lr_decay_updates = [u if u < total_updates else max(1, total_updates - 1) for u in lr_decay_updates]
    lr_decay_updates = sorted(set(lr_decay_updates))

    max_updates = total_updates

    status(f"Updates/epoch (ceil(len(dl_train)/accum)): {updates_per_epoch}")
    status(f"Total updates (EPOCHS*updates_per_epoch): {total_updates}")
    status(f"LR decay updates (por %): {lr_decay_updates}")
    status(f"MAX_UPDATES (ajustado): {max_updates}")

    if PREFETCH_FIRST_BATCH:
        prefetch_one_batch(dl_train, device, name="train")
        prefetch_one_batch(dl_val, device, name="val")

    with Stage("Construyendo modelo ResNet50 CUSTOM"):
        model = build_model(num_classes=num_classes).to(device)
        nparams = sum(p.numel() for p in model.parameters())
        status(f"Modelo listo ✓  Parámetros: {nparams:,}")

        ema = None
        if EMA_DECAY and EMA_DECAY > 0.0:
            ema = ModelEMA(model, decay=EMA_DECAY)
            status(f"EMA activada ✓ decay={EMA_DECAY}")

    with Stage("Construyendo criterion/optimizer/scheduler"):
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = build_optimizer(model)

        if SCHEDULER.lower() != "iter_step":
            raise ValueError("Este script solo soporta SCHEDULER='iter_step' (recipe).")

        scheduler = IterStepLRScheduler(
            base_lr=LR,
            milestones=lr_decay_updates,
            gamma=LR_DECAY_GAMMA,
            max_updates=max_updates,
        )

        status(f"Optimizer: {OPTIMIZER} | LR(base)={LR:.6f} | WD={WEIGHT_DECAY} | mom={MOMENTUM}")
        status(f"Scheduler: iter_step | milestones={lr_decay_updates} | gamma={LR_DECAY_GAMMA} | MAX_UPDATES={max_updates}")

    with Stage("Inicializando AMP GradScaler"):
        scaler = make_grad_scaler(amp_enabled, device=device)
        status("GradScaler listo ✓")

    with Stage("Preparando carpeta de resultados"):
        out_dir = Path(OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        status(f"Resultados en: {out_dir.resolve()}")

    with Stage("Escribiendo config.json"):
        config_dump = {
            "IMAGENET_ROOT": IMAGENET_ROOT,
            "SUBSET_MODE": SUBSET_MODE,
            "SUBSET_K": SUBSET_K,
            "SUBSET_SYNSETS": SUBSET_SYNSETS,
            "SUBSET_NAME": SUBSET_NAME,
            "ARCH": ARCH,
            "ACTIVATION": ACTIVATION,
            "ZERO_INIT_RESIDUAL": ZERO_INIT_RESIDUAL,

            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "EFFECTIVE_BATCH": EFFECTIVE_BATCH,

            "NUM_WORKERS": NUM_WORKERS,
            "PIN_MEMORY": PIN_MEMORY,
            "PERSISTENT_WORKERS": PERSISTENT_WORKERS,

            "OPTIMIZER": OPTIMIZER,
            "BASE_LR_256": BASE_LR_256,
            "LR": LR,
            "MOMENTUM": MOMENTUM,
            "WEIGHT_DECAY": WEIGHT_DECAY,

            "SCHEDULER": SCHEDULER,
            "LR_DECAY_FRACTIONS": LR_DECAY_FRACTIONS,
            "LR_DECAY_UPDATES": lr_decay_updates,
            "LR_DECAY_GAMMA": LR_DECAY_GAMMA,
            "UPDATES_PER_EPOCH": updates_per_epoch,
            "TOTAL_UPDATES": total_updates,
            "MAX_UPDATES": max_updates,

            "NORM_MEAN": NORM_MEAN,
            "NORM_STD": NORM_STD,
            "LABEL_SMOOTHING": LABEL_SMOOTHING,
            "USE_RANDAUGMENT": USE_RANDAUGMENT,
            "RA_NUM_OPS": RA_NUM_OPS,
            "RA_MAGNITUDE": RA_MAGNITUDE,
            "USE_AUTOAUGMENT": USE_AUTOAUGMENT,
            "EMA_DECAY": EMA_DECAY,

            "AMP_ENABLED": AMP_ENABLED,
            "SEED": SEED,
            "DETERMINISTIC": DETERMINISTIC,
            "PREFETCH_FIRST_BATCH": PREFETCH_FIRST_BATCH,
        }
        config_dump["run_id"] = make_run_id(config_dump)
        (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")
        status(f"run_id de esta ejecución: {config_dump['run_id']}")

    # -------------------------
    # AUTO-RESUME + MENSAJE
    # -------------------------
    start_epoch = 1
    best_global_acc = -1.0
    global_update_step = 0

    final_ckpt = out_dir / f"ckpt_epoch_{EPOCHS:03d}.pt"
    if AUTO_RESUME and final_ckpt.exists():
        msg = f"Detectado entrenamiento ya completado: existe {final_ckpt.name} (EPOCHS={EPOCHS})."
        if EXIT_IF_ALREADY_FINISHED:
            status(msg + " Saliendo sin entrenar.")
            return
        else:
            status(msg + " (EXIT_IF_ALREADY_FINISHED=False) => empezaré desde epoch 1.")
            start_epoch = 1

    if AUTO_RESUME and (not final_ckpt.exists()):
        ckpt_path, ckpt_epoch, reason = pick_resume_checkpoint(
            out_dir=out_dir,
            expected_run_id=config_dump["run_id"],
            strict=AUTO_RESUME_STRICT,
        )

        if ckpt_path is None:
            status("No se han detectado checkpoints, entrenando desde 0.")
            start_epoch = 1
            best_global_acc = -1.0
            global_update_step = 0
            scheduler.global_update_step = 0
        else:
            status(f"{reason}, reanudando entrenamiento desde epoch {int(ckpt_epoch) + 1} hasta epoch {EPOCHS}.")

            with Stage("Auto-resume: cargando estados (modelo/optimizer/scheduler/scaler)"):
                start_epoch, best_global_acc, global_update_step = load_checkpoint_for_resume(
                    ckpt_path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    device=device,
                    expected_run_id=config_dump["run_id"],
                    strict=AUTO_RESUME_STRICT,
                    current_classes=classes,
                )

            scheduler.global_update_step = int(global_update_step)

            if start_epoch > EPOCHS:
                status(f"AVISO: start_epoch={start_epoch} > EPOCHS={EPOCHS}. No hay epochs por ejecutar.")
                return
    else:
        status("AUTO_RESUME=False: entrenando desde 0 (ignorando checkpoints existentes).")
        start_epoch = 1
        best_global_acc = -1.0
        global_update_step = 0
        scheduler.global_update_step = 0

    status("Preparación completada. Iniciando entrenamiento...")

    k1, k5, k10 = topk_list_for_num_classes(num_classes)
    s_top1, s_top5, s_top10 = fmt_metric_labels(k1, k5, k10)

    # -------------------------
    # TRAIN LOOP + Ctrl+C
    # -------------------------
    last_completed_epoch = start_epoch - 1
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            current_epoch = epoch

            if scheduler.global_update_step >= scheduler.max_updates:
                status(f"MAX_UPDATES alcanzado ({scheduler.global_update_step}). Parando entrenamiento.")
                break

            t0 = time.time()

            tr_loss, tr_t1, tr_t5, tr_t10, global_update_step = train_one_epoch(
                model=model,
                loader=dl_train,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                scaler=scaler,
                epoch=epoch,
                num_classes=num_classes,
                amp_enabled=amp_enabled,
                ema=ema,
            )

            eval_model = ema.ema if ema is not None else model

            val_loss, val_t1, val_t5, val_t10, global_acc = validate(
                model=eval_model,
                loader=dl_val,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_classes=num_classes,
                amp_enabled=amp_enabled,
            )

            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"[epoch {epoch:03d}] time={dt/60:.2f} min | "
                f"train loss={tr_loss:.4f} | {s_top1}={tr_t1:.2f}% | {s_top5}={tr_t5:.2f}% | {s_top10}={tr_t10:.2f}% | "
                f"val loss={val_loss:.4f} | {s_top1}={val_t1:.2f}% | {s_top5}={val_t5:.2f}% | {s_top10}={val_t10:.2f}% | "
                f"global_accuracy (val, aciertos/total)={global_acc:.2f}% | "
                f"lr={lr_now:.6f} | updates={scheduler.global_update_step}/{scheduler.max_updates}",
                flush=True
            )

            improved = global_acc > best_global_acc
            if improved:
                best_global_acc = global_acc

            state = {
                "epoch": epoch,
                "global_update_step": int(global_update_step),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "best_global_accuracy": best_global_acc,
                "classes": classes,
                "config": config_dump,
                "metrics": {
                    "train": {"loss": tr_loss, "top1": tr_t1, "top5": tr_t5, "top10": tr_t10},
                    "val": {"loss": val_loss, "top1": val_t1, "top5": val_t5, "top10": val_t10, "global_accuracy": global_acc},
                },
                "ema": (ema.state_dict() if ema is not None else None),
            }

            if SAVE_EVERY_EPOCH:
                save_checkpoint(out_dir / f"ckpt_epoch_{epoch:03d}.pt", state)

            if improved:
                save_checkpoint(out_dir / "ckpt_best.pt", state)
                print(f"** Nuevo mejor global_accuracy (val, aciertos/total): {best_global_acc:.2f}% (guardado ckpt_best.pt)", flush=True)

            last_completed_epoch = epoch

        print("Entrenamiento terminado.", flush=True)
        print(f"Mejor global_accuracy val (aciertos/total): {best_global_acc:.2f}%", flush=True)
        print(f"Updates totales: {scheduler.global_update_step}/{scheduler.max_updates}", flush=True)
        print(f"Outputs en: {out_dir.resolve()}", flush=True)

    except KeyboardInterrupt:
        print("", flush=True)
        status("KeyboardInterrupt detectado (Ctrl+C). Deteniendo entrenamiento de forma segura...")

        if SAVE_INTERRUPT_CKPT:
            interrupted_state = {
                "epoch": int(last_completed_epoch),
                "interrupted_at_epoch": int(current_epoch),
                "interrupted_timestamp": datetime.now().isoformat(timespec="seconds"),
                "interrupt_reason": "KeyboardInterrupt (Ctrl+C)",
                "global_update_step": int(scheduler.global_update_step),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "best_global_accuracy": best_global_acc,
                "classes": classes,
                "config": config_dump,
                "ema": (ema.state_dict() if ema is not None else None),
            }
            save_checkpoint(out_dir / INTERRUPT_CKPT_NAME, interrupted_state)
            status(
                f"Guardado {INTERRUPT_CKPT_NAME} ✓ "
                f"(epoch={last_completed_epoch}, reanudará en epoch={last_completed_epoch+1} hasta EPOCHS={EPOCHS})."
            )

        status("Salida limpia tras Ctrl+C.")
        return


if __name__ == "__main__":
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_f = None
    old_out, old_err = sys.stdout, sys.stderr

    try:
        if LOG_TO_FILE:
            mode = "a" if LOG_APPEND else "w"
            log_path = out_dir / LOG_FILENAME
            log_f = open(log_path, mode, buffering=1, encoding="utf-8")
            sys.stdout = Tee(old_out, log_f)
            sys.stderr = Tee(old_err, log_f)
            print(f"[{_ts()}] Logging a: {log_path}", flush=True)

        main()

    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        if log_f is not None:
            log_f.close()
