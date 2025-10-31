cmake_minimum_required(VERSION 3.10)
project(STONEFISH_RL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Stonefish REQUIRED 1.5.0)
find_package(PkgConfig REQUIRED)

file(GLOB_RECURSE SOURCE "${CMAKE_SOURCE_DIR}/src/*.cpp")

pkg_check_modules(ZMQ REQUIRED libzmq)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${ZMQ_INCLUDE_DIRS}
)


add_executable(StonefishRLTest executables/main.cpp )
add_executable(TestSender executables/test_sender.cpp )

target_sources(StonefishRLTest PRIVATE
    ${SOURCE} 
)

target_sources(TestSender PRIVATE
    ${SOURCE} 
)

# Link with Stonefish library
target_link_libraries(StonefishRLTest
    Stonefish::Stonefish
    ${ZMQ_LIBRARIES}
)

target_link_libraries(TestSender
    Stonefish::Stonefish
    ${ZMQ_LIBRARIES}
)