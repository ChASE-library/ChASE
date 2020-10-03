set(subdirs
    0_hello_world
    1_sequence_eigenproblems
    2_input_output
)

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()


