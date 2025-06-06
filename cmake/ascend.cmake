execute_process(
    COMMAND python -c "from torch.utils import cmake_prefix_path; \
    print(cmake_prefix_path + '/Torch', end='')"
    OUTPUT_VARIABLE Torch_DIR
)

execute_process(
    COMMAND python -c "from importlib.metadata import distribution; \
    print(str(distribution('torch_npu').locate_file('torch_npu')), end='')"
    OUTPUT_VARIABLE Torch_npu_ROOT
)

execute_process(
    COMMAND python -c "import torch; \
    print('1' if torch.compiled_with_cxx11_abi() else '0', end='')"
    OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
)

execute_process(
    COMMAND python -c "import torch;  from packaging import version; \
    torch_version = version.parse(torch.__version__).base_version; \
    print('1' if version.parse(torch_version) > version.parse('2.3.1') else '0', end='')"
    OUTPUT_VARIABLE Torch_npu_VERSION_HIGHER_THAN_231
)

find_package(Torch REQUIRED)
find_package(Torch_npu REQUIRED)
find_package(CANNToolkit REQUIRED)
find_package(ATB)
