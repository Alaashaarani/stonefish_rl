cmake_minimum_required(VERSION 3.10)
project(STONEFISH_RL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Stonefish REQUIRED 1.5.0)
find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB_RECURSE SOURCE "${CMAKE_SOURCE_DIR}/src/*.cpp")

pkg_check_modules(ZMQ REQUIRED libzmq)
# pkg_check_modules(yaml-cpp REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${ZMQ_INCLUDE_DIRS}
)


add_executable(StonefishRL src/cpp/main.cpp )

target_sources(StonefishRL PRIVATE
    ${SOURCE} 
)

# Link with Stonefish library
target_link_libraries(StonefishRL
    Stonefish::Stonefish
    ${ZMQ_LIBRARIES}
    # ${nlohmann_json_LIBRARIES}
    yaml-cpp
)

# target_link_libraries(TestSender
#     Stonefish::Stonefish
#     ${ZMQ_LIBRARIES}
# )
