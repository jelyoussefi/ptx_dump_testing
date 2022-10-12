cmake_minimum_required(VERSION 3.5.1)

project(matrix_ops VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang )

	if (CUDA)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda -Wno-unknown-cuda-version ")
		else()
			
		endif(CUDA)	
	
endif()


add_executable(${PROJECT_NAME}  ./main.cpp ./matrix_mult.cpp)
        
install(TARGETS ${PROJECT_NAME}  RUNTIME DESTINATION bin)
