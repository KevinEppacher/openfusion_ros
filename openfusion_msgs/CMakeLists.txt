cmake_minimum_required(VERSION 3.10)
project(openfusion_msgs)

# Suppress Python policy warnings from ROS2 build chain
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

############################################################
# Dependencies
############################################################
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

############################################################
# Interface generation
############################################################
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SaveSemanticMap.srv"
  DEPENDENCIES
    std_msgs
)

############################################################
# Export
############################################################
ament_export_dependencies(
  rosidl_default_runtime
  std_msgs
)

ament_package()
