find_package(Doxygen REQUIRED)

#Doxygen
set(DOXYGEN_INPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
message("DOXYGEN_INPUT_DIRECTORY = ${DOXYGEN_INPUT_DIRECTORY}")

set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
message("DOXYGEN_OUTPUT_DIRECTORY = ${DOXYGEN_OUTPUT_DIRECTORY}")

set(DOXYGEN_HTML_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html)
message("DOXYGEN_HTML_OUTPUT = ${DOXYGEN_HTML_OUTPUT}")

set(DOXYGEN_EXCLUDE_DIRS "${CMAKE_SOURCE_DIR}/build")
#set(DOXYGEN_EXCLUDE_DIRS "${CMAKE_SOURCE_DIR}/tests")
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIRECTORY}/xml/index.xml)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_PREDEFINED "HAS_CUDA" "HAS_SCALAPACK" "USE_MPI" "HAS_NCCL" "ENABLE_MIXED_PRECISION" "CHASE_OUTPUT")
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_INTERNAL_DOCS NO)
set(DOXYGEN_EXCLUDE_PATTERNS "*.rst")
set(DOXYGEN_EXCLUDE_PATTERNS "*.tex")
set(DOXYGEN_EXCLUDE_PATTERNS "*/build*")
set(DOXYGEN_PROJECT_BRIEF "ChASE: an Iterative Solver for Dense Eigenproblems")
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_TEMPLATE_RELATIONS YES)
set(DOXYGEN_OPTIMIZE_FOR_FORTRAN YES)
set(DOXYGEN_INLINE_INHERITED_MEMBERS YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_CLASS_DIAGRAMS YES)
#set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
#set(DOXYGEN_HAVE_DOT YES)
set(DOXYGEN_CLASS_GRAPH YES)
set(DOXYGEN_COLLABORATION_GRAPH YES)
#set(DOXYGEN_UML_LOOK YES)
#set(DOXYGEN_UML_LIMIT_NUM_FIELDS 50)
set(DOXYGEN_TEMPLATE_RELATIONS YES)
#set(DOXYGEN_DOT_GRAPH_MAX_NODES 100)
#set(DOXYGEN_MAX_DOT_GRAPH_DEPTH 0)
#set(DOXYGEN_DOT_TRANSPARENT YES)
#set(DOXYGEN_DOT_IMAGE_FORMAT "svg")
#set(DOXYGEN_INTERACTIVE_SVG YES)


make_directory(${DOXYGEN_HTML_OUTPUT})

doxygen_add_docs(Doxygen
    ${DOXYGEN_INPUT_DIRECTORY}
    DOXYGEN_PROJECT_NAME ${CMAKE_PROJECT_NAME}
    DOXYGEN_PROJECT_NUMBER ${CMAKE_PROJECT_VERSION}
    COMMENT "Generating API documentation with Doxygen")

#Sphinx
find_package(Python REQUIRED)

#find sphinx
message( ${Python_EXECUTABLE})
execute_process(COMMAND ${Python_EXECUTABLE} -c "import sphinx"  RESULT_VARIABLE sphinx_ret)

if (sphinx_ret EQUAL "0")
  set(Sphinx_FOUND 1)
else()
  set(Sphinx_FOUND 0)
endif()

if(Sphinx_FOUND)
    message(STATUS "Sphinx is FOUND")
else()
    message(SEND_ERROR "Could not find Sphinx for Documentation")
endif()

#find breathe
set(PYTHON_DEP_STRING "Please install Python package breathe (at least version 4.18.0) via: pip install breathe")

execute_process(COMMAND ${Python_EXECUTABLE} -c "from packaging import version; import sys; import breathe; sys.exit(version.parse(\"4.18.0\") > version.parse(breathe.__version__))" RESULT_VARIABLE ret)
if (ret)
  message(FATAL_ERROR "Missing Documentation dependency breathe. ${PYTHON_DEP_STRING}")
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/sphinx)
set(SPHINX_HTML_INDEX_FILE ${SPHINX_BUILD}/html/index.html)

#Sphinx HTML
add_custom_target(Sphinx ALL
                  COMMAND  ${Python_EXECUTABLE} -m sphinx -b html
                  -Dbreathe_projects.ChASE=${DOXYGEN_OUTPUT_DIRECTORY}/xml
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}/html
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS Doxygen
                  COMMENT "Generating HTML documentation with Sphinx into {CMAKE_BINARY_DIR}/sphinx/")