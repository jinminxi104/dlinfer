from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any


class AclDataType(IntEnum):
    ACL_DT_UNDEFINED = 0


@dataclass
class LinearParam:
    transposeA: bool = False
    transposeB: bool = True
    hasBias: bool = True


class ElewiseType(IntEnum):
    ELEWISE_UNDEFINED = 0
    ELEWISE_CAST = 1
    ELEWISE_MULS = 2
    ELEWISE_COS = 3
    ELEWISE_SIN = 4
    ELEWISE_NEG = 5
    ELEWISE_QUANT = 6
    ELEWISE_LOGICAL_NOT = 7
    ELEWISE_ADD = 8
    ELEWISE_MUL = 9
    ELEWISE_REALDIV = 10
    ELEWISE_LOGICAL_AND = 11
    ELEWISE_LOGICAL_OR = 12
    ELEWISE_LESS = 13
    ELEWISE_GREATER = 14
    ELEWISE_SUB = 15
    ELEWISE_EQUAL = 16
    ELEWISE_QUANT_PER_CHANNEL = 17
    ELEWISE_DEQUANT_PER_CHANNEL = 18
    ELEWISE_DYNAMIC_QUANT = 19
    ELEWISE_TANH = 20


class ActivationType(Enum):
    ACTIVATION_UNDEFINED = 0
    ACTIVATION_RELU = auto()
    ACTIVATION_GELU = auto()
    ACTIVATION_FAST_GELU = auto()
    ACTIVATION_SWISH = auto()
    ACTIVATION_LOG = auto()
    ACTIVATION_SWIGLU_FORWARD = auto()
    ACTIVATION_SWIGLU_BACKWARD = auto()
    ACTIVATION_MAX = auto()


class QuantType(Enum):
    QUANT_UNDEFINED = 0
    QUANT_INT4 = auto()
    QUANT_INT8 = auto()
    QUANT_INT16 = auto()
    QUANT_FLOAT8 = auto()
    QUANT_FLOAT16 = auto()


@dataclass
class ElewiseQuantParam:
    inputScale: float = 1.0
    asymmetric: bool = False
    inputOffset: int = 0


@dataclass
class MulsParam:
    varAttr: float = 0.0


@dataclass
class ElewiseParam:
    elewiseType: ElewiseType = ElewiseType.ELEWISE_UNDEFINED
    quantParam: ElewiseQuantParam = field(default_factory=ElewiseQuantParam)
    mulsParam: MulsParam = field(default_factory=MulsParam)
    outTensorType: AclDataType = AclDataType.ACL_DT_UNDEFINED


@dataclass
class AclNnCastParam:
    name: str = ""
    outTensorType: AclDataType = AclDataType.ACL_DT_UNDEFINED


class RmsNormType(IntEnum):
    RMS_NORM_UNDEFINED = 0
    RMS_NORM_NORM = 1
    RMS_NORM_PRENORM = 2
    RMS_NORM_POSTNORM = 3


class PrecisionMode(IntEnum):
    HIGH_PRECISION_MODE = 0
    HIGH_PERFORMANCE_MODE = 1


class ModelType(IntEnum):
    LLAMA_MODEL = 0
    GEMMA_MODEL = 1


class QuantType(IntEnum):
    QUANT_UNDEFINED = 0


class DynamicQuantType(IntEnum):
    DYNAMIC_QUANT_UNDEFINED = 0


@dataclass
class NormParam:
    quantType: QuantType = QuantType.QUANT_UNDEFINED
    epsilon: float = 1e-5
    layerNormEps: float = 1e-5
    rstd: bool = False
    precisionMode: PrecisionMode = PrecisionMode.HIGH_PRECISION_MODE
    modelType: ModelType = ModelType.LLAMA_MODEL
    dynamicQuantType: DynamicQuantType = DynamicQuantType.DYNAMIC_QUANT_UNDEFINED


@dataclass
class PreNormParam:
    quantType: QuantType = QuantType.QUANT_UNDEFINED
    epsilon: float = 1e-5
    hasBias: bool = False


@dataclass
class PostNormParam:
    quantType: QuantType = QuantType.QUANT_UNDEFINED
    epsilon: float = 1e-5
    hasBias: bool = False


@dataclass
class RmsNormParam:
    layerType: RmsNormType = RmsNormType.RMS_NORM_UNDEFINED
    normParam: NormParam = NormParam()
    preNormParam: PreNormParam = PreNormParam()
    postNormParam: PostNormParam = PostNormParam()


@dataclass
class RopeParam:
    rotaryCoeff: int = 4
    cosFormat: int = 0


class SelfAttentionCalcType(IntEnum):
    UNDEFINED = 0
    ENCODER = 1
    DECODER = 2
    PA_ENCODER = 3


class SelfAttentionKernelType(IntEnum):
    KERNELTYPE_DEFAULT = 0
    KERNELTYPE_HIGH_PRECISION = 1


class SelfAttentionClampType(IntEnum):
    CLAMP_TYPE_UNDEFINED = 0
    CLAMP_TYPE_MIN_MAX = 1


class SelfAttentionMaskType(IntEnum):
    MASK_TYPE_UNDEFINED = 0
    MASK_TYPE_NORM = 1
    MASK_TYPE_ALIBI = 2
    MASK_TYPE_NORM_COMPRESS = 3
    MASK_TYPE_ALIBI_COMPRESS = 4
    MASK_TYPE_ALIBI_COMPRESS_SQRT = 5
    MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 6


@dataclass
class SelfAttentionParam:
    headNum: int = 0
    kvHeadNum: int = 0
    qScale: float = 1.0
    qkScale: float = 1.0
    batchRunStatusEnable: bool = False
    isTriuMask: int = 0
    calcType: SelfAttentionCalcType = SelfAttentionCalcType.UNDEFINED
    kernelType: SelfAttentionKernelType = SelfAttentionKernelType.KERNELTYPE_DEFAULT
    clampType: SelfAttentionClampType = SelfAttentionClampType.CLAMP_TYPE_UNDEFINED
    clampMin: float = 0.0
    clampMax: float = 0.0
    maskType: SelfAttentionMaskType = SelfAttentionMaskType.MASK_TYPE_UNDEFINED


class ReshapeAndCacheCompressType(IntEnum):
    COMPRESS_TYPE_UNDEFINED = 0
    COMPRESS_TYPE_KVHEAD = 1


@dataclass
class ReshapeAndCacheParam:
    compressType: ReshapeAndCacheCompressType = (
        ReshapeAndCacheCompressType.COMPRESS_TYPE_UNDEFINED
    )


class PagedAttentionMaskType(IntEnum):
    UNDEFINED = 0
    MASK_TYPE_NORM = 1
    MASK_TYPE_ALIBI = 2
    MASK_TYPE_SPEC = 3


class PagedAttentionQuantType(IntEnum):
    TYPE_QUANT_UNDEFINED = 0
    TYPE_DEQUANT_FUSION = 1


class PagedAttentionCompressType(IntEnum):
    COMPRESS_TYPE_UNDEFINED = 0
    COMPRESS_TYPE_KVHEAD = 1


class PagedAttentionCalcType(IntEnum):
    CALC_TYPE_UNDEFINED = 0
    CALC_TYPE_SPEC = 1


@dataclass
class PagedAttentionParam:
    headNum: int = 0
    qkScale: float = 1.0
    kvHeadNum: int = 0
    maskType: PagedAttentionMaskType = PagedAttentionMaskType.UNDEFINED
    batchRunStatusEnable: bool = False
    quantType: PagedAttentionQuantType = PagedAttentionQuantType.TYPE_QUANT_UNDEFINED
    hasQuantOffset: bool = False
    compressType: PagedAttentionCompressType = (
        PagedAttentionCompressType.COMPRESS_TYPE_UNDEFINED
    )
    calcType: PagedAttentionCalcType = PagedAttentionCalcType.CALC_TYPE_UNDEFINED


@dataclass
class AddRmsNormParam:
    epsilon: float = 1.0


@dataclass
class TransposeParam:
    name: str
    perm: list[int]


@dataclass
class SplitParam:
    splitDim: int = 0
    splitNum: int = 2
    splitSizes: list[int] = field(default_factory=list)


@dataclass
class MlpQuantParam:
    quantType: QuantType = QuantType.QUANT_UNDEFINED
    elewiseType: ElewiseType = ElewiseType.ELEWISE_UNDEFINED
    inputScale: float = 1.0
    inputOffset: int = 0
    tilingN: int = 0
    tilingK: int = 0
    isQuantOp: bool = False


@dataclass
class MlpCommParam:
    rank: int = 0
    rankSize: int = 1
    rankRoot: int = 0
    hcclComm: Any = None
    backend: str = "hccl"


@dataclass
class MlpGateParamV2:
    activationType: ActivationType = ActivationType.ACTIVATION_UNDEFINED
    transposeB: bool = True
    isBias: bool = False
    isPack: bool = False
    isQuant: bool = False
    isSparse: bool = False
    noGate: bool = False
    isBF16: bool = False
    commDownParam: MlpCommParam = MlpCommParam()
    quantUpParam: MlpQuantParam = MlpQuantParam()
    quantGateParam: MlpQuantParam = MlpQuantParam()
    quantDownParam: MlpQuantParam = MlpQuantParam()


class GeLUMode(IntEnum):
    TANH_MODE = 0
    NONE_MODE = 1


@dataclass
class ConcatParam:
    concatDim: int = 0


@dataclass
class AclNnConcatParam:
    name: str = ""
    concatDim: int = 0
    inputNum: int = 0


@dataclass
class ActivationParam:
    activationType: str = "ACTIVATION_UNDEFINED"
    scale: float = 1.0  # for Swish
    dim: int = -1  # for Swiglu
    geluMode: GeLUMode = GeLUMode.TANH_MODE


@dataclass
class BatchMatMulParam:
    name: str = ""
    cubeMathType: int = 1


@dataclass
class GatherParam:
    name: str = ""
    axis: int = 0
    batchDims: int = 0


def custom_asdict_factory(data):
    def convert_value(obj):
        if isinstance(obj, IntEnum):
            return obj.value
        return obj

    return {k: convert_value(v) for k, v in data}


def to_dict(data):
    return asdict(data, dict_factory=custom_asdict_factory)