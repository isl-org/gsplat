cmake_minimum_required(VERSION 3.23...4.0)  # Need min 3.23 on Windows

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icx)

project(gsplat_sycl)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (NOT SYCL_AOT_TARGETS)
        set (SYCL_AOT_TARGETS "spir64" CACHE STRING "Comma separated list of SYCL targets for ahead of time compilation. See https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md for a full list.")
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "
import sys, os, glob
try:
    import torch
    from torch.utils import cpp_extension
    import pybind11
except ImportError:
    sys.exit(1)

lib_paths = cpp_extension.library_paths(True)
lib_name = 'libtorch_python.so*' if sys.platform != 'win32' else 'torch_python.lib'
base_path = os.path.join(lib_paths[0], lib_name)
matches = glob.glob(base_path)
torch_python_lib = matches[0] if matches else ''

print(f'{torch.utils.cmake_prefix_path};{torch_python_lib};{pybind11.get_cmake_dir()}')
"
  OUTPUT_VARIABLE PYTHON_CONFIG_LIST
  RESULT_VARIABLE PYTHON_CONFIG_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYTHON_CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to retrieve configuration from Python. Ensure torch and pybind11 are installed.")
endif()

list(LENGTH PYTHON_CONFIG_LIST LIST_LEN)
if(LIST_LEN LESS 3)
    message(FATAL_ERROR "Python script returned incomplete information.")
endif()
list(GET PYTHON_CONFIG_LIST 0 Torch_DIR_From_Python)
list(GET PYTHON_CONFIG_LIST 1 TORCH_PYTHON_LIB)
list(GET PYTHON_CONFIG_LIST 2 PYBIND11_CMAKE_DIR)

if(Torch_DIR_From_Python AND IS_DIRECTORY "${Torch_DIR_From_Python}")
  set(Torch_DIR ${Torch_DIR_From_Python})
  message(STATUS "Found Torch CMake directory via Python: ${Torch_DIR}")
  find_package(Torch REQUIRED HINTS ${Torch_DIR_From_Python})
else()
  message(FATAL_ERROR "Could not find Torch via Python introspection.")
endif()

if (NOT EXISTS "${TORCH_PYTHON_LIB}")
    message(FATAL_ERROR "Could not find torch_python library.")
else()
    message(STATUS "Found torch_python library at: ${TORCH_PYTHON_LIB}")
endif()

message(STATUS "Found pybind11 in ${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED HINTS ${PYBIND11_CMAKE_DIR} "${Python_SITELIB}/pybind11/share/cmake/pybind11")

set(SYCL_SOURCES
  ext.cpp
  src/adam.cpp
  src/intersect_offset.cpp
  src/intersect_tile.cpp
  src/null.cpp
  src/projection_2dgs_fused_bwd.cpp
  src/projection_2dgs_fused_fwd.cpp
  src/projection_2dgs_packed_bwd.cpp
  src/projection_2dgs_packed_fwd.cpp
  src/projection_ewa_3dgs_fused_bwd.cpp
  src/projection_ewa_3dgs_fused_fwd.cpp
  src/projection_ewa_3dgs_packed_bwd.cpp
  src/projection_ewa_3dgs_packed_fwd.cpp
  src/projection_ewa_simple_bwd.cpp
  src/projection_ewa_simple_fwd.cpp
  src/projection_ut_3dgs_fused.cpp
  src/quat_scale_to_covar_preci_bwd.cpp
  src/quat_scale_to_covar_preci_fwd.cpp
  src/rasterize_to_indices_2dgs.cpp
  src/rasterize_to_indices_3dgs.cpp
  src/rasterize_to_pixels_2dgs_bwd.cpp
  src/rasterize_to_pixels_2dgs_fwd.cpp
  src/rasterize_to_pixels_3dgs_bwd.cpp
  src/rasterize_to_pixels_3dgs_fwd.cpp
  src/rasterize_to_pixels_from_world_3dgs_bwd.cpp
  src/rasterize_to_pixels_from_world_3dgs_fwd.cpp
  src/relocation.cpp
  src/spherical_harmonics_bwd.cpp
  src/spherical_harmonics_fwd.cpp
)

set(SYCL_MODULE_NAME gsplat_sycl_kernels)

pybind11_add_module(${SYCL_MODULE_NAME} MODULE ${SYCL_SOURCES})

target_include_directories( ${SYCL_MODULE_NAME} SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/csrc/third_party/glm 
)

target_compile_definitions(${SYCL_MODULE_NAME} PRIVATE 
    TORCH_EXTENSION_NAME=${SYCL_MODULE_NAME}
)

target_compile_options(${SYCL_MODULE_NAME} PRIVATE -fsycl)
target_compile_features(${SYCL_MODULE_NAME} PUBLIC cxx_std_17)

target_link_options(${SYCL_MODULE_NAME} PRIVATE -fsycl -fsycl-targets=${SYCL_AOT_TARGETS})
target_link_libraries(${SYCL_MODULE_NAME} PRIVATE torch ${TORCH_PYTHON_LIB})


# Fix for icx: error: '-MP' is not supported with offloading enabled
if (WIN32 AND NOT UNIX)
    get_target_property(CURRENT_OPTIONS ${SYCL_MODULE_NAME} COMPILE_OPTIONS)
    string(REPLACE "/MP" "" MODIFIED_OPTIONS "${CURRENT_OPTIONS}")
    set_target_properties(${SYCL_MODULE_NAME} PROPERTIES COMPILE_OPTIONS "${MODIFIED_OPTIONS}")
endif ()
if (UNIX AND NOT APPLE)
   # Find libtorch_xpu and libsycl at runtime in Python environment
   set_target_properties(${SYCL_MODULE_NAME} PROPERTIES INSTALL_RPATH 
           "$ORIGIN/../../torch/lib/;$ORIGIN/../../../../")
endif()