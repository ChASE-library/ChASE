# -*- Mode: cmake -*-

# Specify the minimum version for CMake
cmake_minimum_required( VERSION 3.7 )

# for the moment ChASE only supports Elemental v.84
# which does not come with CMake support
if(EXISTS "$ENV{ELEMENTAL_ROOT}/conf/ElemVar")

  add_library( chase_elemental INTERFACE )
  target_link_libraries(chase_elemental INTERFACE chase_algorithm)

  target_include_directories( chase_elemental INTERFACE
    "$ENV{ELEMENTAL_ROOT}/include/"
    )

  target_link_libraries( chase_elemental INTERFACE
    "-L/homeb/slai/slai10/elemental/lib -lelemental -lpmrrr -lpthread"
    )
endif()
