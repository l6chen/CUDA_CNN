include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS 
../cnn/*.h
../cnn/*.cuh
../cnn/*.cu
../cnn/*.cpp
)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS} )
    cuda_add_executable(unittest
      ${HEADERS}
      test_main.cu

      OPTIONS --cudart=shared  -arch sm_50 -use_fast_math
      DEBUG -Xptxas -v -g -G 
      )

      target_link_libraries( 
        unittest
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CURAND_LIBRARY}
        ${OpenCV_LIBS}
      )
endif()
