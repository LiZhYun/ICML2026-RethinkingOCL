from slotcontrast.modules import timm
from slotcontrast.modules.decoders import build as build_decoder
from slotcontrast.modules.dynamics import build as build_dynamics_predictor
from slotcontrast.modules.encoders import build as build_encoder
from slotcontrast.modules.groupers import build as build_grouper
from slotcontrast.modules.initializers import build as build_initializer
from slotcontrast.modules.memory import MemoryBank, SlotMemoryEncoder
from slotcontrast.modules.memory import build_memory_bank, build_memory_encoder
from slotcontrast.modules.networks import build as build_network
from slotcontrast.modules.utils import Resizer, SoftToHardMask
from slotcontrast.modules.utils import build as build_utils
from slotcontrast.modules.utils import build_module, build_torch_function, build_torch_module
from slotcontrast.modules.amodal_particles import AmodalParticleProcessor
from slotcontrast.modules.amodal_particles import build as build_amodal_particles
from slotcontrast.modules.video import LatentProcessor, MapOverTime, ScanOverTime
from slotcontrast.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_dynamics_predictor",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_memory_bank",
    "build_memory_encoder",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "ScanOverTime",
    "LatentProcessor",
    "AmodalParticleProcessor",
    "MemoryBank",
    "SlotMemoryEncoder",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "dynamics_predictors": build_dynamics_predictor,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "initializers": build_initializer,
    "memory_banks": build_memory_bank,
    "memory_encoders": build_memory_encoder,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
    "amodal_particles": build_amodal_particles,
}
