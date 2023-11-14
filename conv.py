from rknn.api import RKNN

INPUT_MODEL = "yolov8x.onnx"
WIDTH = 320
HEIGHT = 320
OUTPUT_MODEL_BASENAME = 'yolov8x'
QUANTIZATION = False
DATASET = './dataset_coco10.txt'

# Config
MEAN_VALUES = [[0, 0, 0]]
STD_VALUES = [[255, 255, 255]]
QUANT_IMG_RGB2BGR = True
QUANTIZED_DTYPE = "asymmetric_quantized-8"
QUANTIZED_ALGORITHM = "normal"
QUANTIZED_METHOD = "channel"
FLOAT_DTYPE = "float16"
OPTIMIZATION_LEVEL = 2
TARGET_PLATFORM = "rk3588"
CUSTOM_STRING = None
REMOVE_WEIGHT = None
COMPRESS_WEIGHT = False
SINGLE_CORE_MODE = False
MODEL_PRUNNING = False
OP_TARGET = None
DYNAMIC_INPUT = None


OUTPUT_MODEL = OUTPUT_MODEL_BASENAME + '-' + str(WIDTH) + 'x' + str(HEIGHT) + ".rknn"

rknn = RKNN()
rknn.config(mean_values=MEAN_VALUES,
            std_values=STD_VALUES,
            quant_img_RGB2BGR=QUANT_IMG_RGB2BGR,
            quantized_dtype=QUANTIZED_DTYPE,
            quantized_algorithm=QUANTIZED_ALGORITHM,
            quantized_method=QUANTIZED_METHOD,
            float_dtype=FLOAT_DTYPE,
            optimization_level=OPTIMIZATION_LEVEL,
            target_platform=TARGET_PLATFORM,
            custom_string=CUSTOM_STRING,
            remove_weight=REMOVE_WEIGHT,
            compress_weight=COMPRESS_WEIGHT,
            single_core_mode=SINGLE_CORE_MODE,
            model_pruning=MODEL_PRUNNING,
            op_target=OP_TARGET,
            dynamic_input=DYNAMIC_INPUT)

# if rknn.load_pytorch("./input/" + INPUT_MODEL, [[HEIGHT, WIDTH, 3]]) != 0:
if rknn.load_onnx("./input/" + INPUT_MODEL) != 0:
    print('Error loading model.')
    exit()

if rknn.build(do_quantization=QUANTIZATION, dataset=DATASET) != 0:
    print('Error building model.')
    exit()

if rknn.export_rknn("./output/" + OUTPUT_MODEL) != 0:
    print('Error exporting rknn model.')
    exit()