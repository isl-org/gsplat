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
  COMMAND "${Python_EXECUTABLE}" -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE Torch_DIR_From_Python 
  RESULT_VARIABLE _torch_path_result    
  ERROR_QUIET                           
)
if(NOT _torch_path_result EQUAL 0)
  message(WARNING "Failed to get Torch CMake path from Python. "
                  "Make sure PyTorch is installed in the Python environment: ${Python_EXECUTABLE}")
  set(Torch_DIR_From_Python "")
endif()

if(Torch_DIR_From_Python AND IS_DIRECTORY "${Torch_DIR_From_Python}")
  set(Torch_DIR ${Torch_DIR_From_Python})
  message(STATUS "Found Torch CMake directory via Python: ${Torch_DIR}")
  find_package(Torch REQUIRED HINTS ${Torch_DIR_From_Python})
else()
  message(FATAL_ERROR "Could not find Torch via Python introspection. "
                      "Please ensure PyTorch is installed or set CMAKE_PREFIX_PATH/Torch_DIR manually.")
endif()

string(CONCAT TORCH_PYTHON_LIB_NAME ${CMAKE_SHARED_LIBRARY_PREFIX} "torch_python" ${CMAKE_LINK_LIBRARY_SUFFIX})
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import os; from torch.utils import cpp_extension; print(os.path.join(cpp_extension.library_paths(True)[0], '${TORCH_PYTHON_LIB_NAME}'))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PYTHON_LIB
)

# Add a fallback to find the file if it exists with a different extension
if (NOT EXISTS "${TORCH_PYTHON_LIB}")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import os, glob; from torch.utils import cpp_extension; base_path = os.path.join(cpp_extension.library_paths(True)[0], '${CMAKE_SHARED_LIBRARY_PREFIX}torch_python'); matches = glob.glob(base_path + '.*'); print(matches[0] if matches else '')"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE TORCH_PYTHON_LIB_FALLBACK
    )
    if (EXISTS "${TORCH_PYTHON_LIB_FALLBACK}")
        set(TORCH_PYTHON_LIB "${TORCH_PYTHON_LIB_FALLBACK}")
        message(STATUS "Found torch_python library using fallback: ${TORCH_PYTHON_LIB}")
    endif()
endif()


if (NOT EXISTS "${TORCH_PYTHON_LIB}")
    message(FATAL_ERROR "Could not find ${TORCH_PYTHON_LIB_NAME} at ${TORCH_PYTHON_LIB}. Please check your PyTorch installation.")
else()
    message(STATUS "Found torch_python library at: ${TORCH_PYTHON_LIB}")
endif()


set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python_SITELIB}/pybind11/share/cmake/pybind11")

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