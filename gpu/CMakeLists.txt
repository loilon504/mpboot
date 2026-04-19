option(USE_OPENMP "Enable OpenMP acceleration for CPU-side loops" OFF)

if(USE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP requested but not found; CPU loops will run single-threaded")
    endif()
endif()

add_library(mpboot_compile_flags INTERFACE)
target_compile_features(mpboot_compile_flags INTERFACE cxx_std_14)
target_compile_options(mpboot_compile_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# CUDA compile flags
if(USE_GPU)
    target_compile_options(mpboot_compile_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    )
    # if(OpenMP_CXX_FOUND)
    #     separate_arguments(MPBOOT_OPENMP_CXX_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
    #     foreach(omp_flag IN LISTS MPBOOT_OPENMP_CXX_FLAGS)
    #         target_compile_options(mpboot_compile_flags INTERFACE
    #             $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${omp_flag}>
    #         )
    #     endforeach()
    # endif()
    target_compile_definitions(mpboot_compile_flags INTERFACE USE_GPU)
endif()

add_library(MPBOOTlib INTERFACE)
file(GLOB_RECURSE MPBOOT_CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
target_sources(MPBOOTlib INTERFACE ${MPBOOT_CPP_FILES})

# Add CUDA sources if use GPU
if(USE_GPU)
    file(GLOB_RECURSE MPBOOT_CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
    target_sources(MPBOOTlib INTERFACE ${MPBOOT_CU_FILES})
    target_link_libraries(MPBOOTlib INTERFACE CUDA::cudart)
endif()

target_include_directories(MPBOOTlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(MPBOOTlib INTERFACE mpboot_compile_flags)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MPBOOTlib INTERFACE OpenMP::OpenMP_CXX)
endif()