find_package(Doxygen REQUIRED)

#python is required for Sphinx
find_package(Python REQUIRED)

#find sphinx
execute_process(COMMAND ${Python_EXECUTABLE} -c "import sphinx;import breathe"  RESULT_VARIABLE sphinx_ret)

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

#Doxygen
set(DOXYGEN_INPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
message("DOXYGEN_INPUT_DIRECTORY = ${DOXYGEN_INPUT_DIRECTORY}")

set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
message("DOXYGEN_OUTPUT_DIRECTORY = ${DOXYGEN_OUTPUT_DIRECTORY}")

set(DOXYGEN_HTML_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html)
message("DOXYGEN_HTML_OUTPUT = ${DOXYGEN_HTML_OUTPUT}")

set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIRECTORY}/xml/index.xml)
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_INTERNAL_DOCS NO)
set(DOXYGEN_EXCLUDE_PATTERNS "*.rst")
set(DOXYGEN_EXCLUDE_PATTERNS "*.tex")
set(DOXYGEN_PROJECT_BRIEF "ChASE: an Iterative Solver for Dense Eigenproblems")

make_directory(${DOXYGEN_HTML_OUTPUT})

doxygen_add_docs(Doxygen
    ${DOXYGEN_INPUT_DIRECTORY}
    DOXYGEN_PROJECT_NAME ${CMAKE_PROJECT_NAME}
    DOXYGEN_PROJECT_NUMBER ${CMAKE_PROJECT_VERSION}
    COMMENT "Generating API documentation with Doxygen")

#Sphinx
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/sphinx)
set(SPHINX_HTML_INDEX_FILE ${SPHINX_BUILD}/html/index.html)

if (UNIX)
  set(separator ":")
else ()
  set(separator ";")
endif()

#Sphinx HTML
add_custom_target(Sphinx ALL
                  COMMAND  ${Python_EXECUTABLE} -m sphinx -b html
                  -Dbreathe_projects.ChASE=${DOXYGEN_OUTPUT_DIRECTORY}/xml
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}/html
                  COMMAND  ${Python_EXECUTABLE} -m sphinx -b latex
                  -Dbreathe_projects.ChASE=${DOXYGEN_OUTPUT_DIRECTORY}/xml
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}/latex
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS Doxygen
                  COMMENT "Generating HTML & Latex documentation with Sphinx into {CMAKE_BINARY_DIR}/sphinx/")

#Generating pdf version of documentation by LATEX
find_package(LATEX)
if(LATEX_FOUND)
  set(LATEX_PDF_FILE ${CMAKE_PROJECT_NAME}_${CMAKE_PROJECT_VERSION}.pdf)
  make_directory(${CMAKE_BINARY_DIR}/sphinx/pdf)
  add_custom_target(PDF ALL
                    COMMAND ${XELATEX_COMPILER} -output-directory ${SPHINX_BUILD}/latex -synctex=1 -interaction=nonstopmode ${SPHINX_BUILD}/latex/*.tex
                    COMMAND ${CMAKE_COMMAND} -E copy *.pdf ${SPHINX_BUILD}/pdf/${LATEX_PDF_FILE}.pdf
                    WORKING_DIRECTORY ${SPHINX_BUILD}/latex
                    DEPENDS Sphinx
                    COMMENT "Generating PDF documentation into ${SPHINX_BUILD}/pdf/")
endif()

install(DIRECTORY ${SPHINX_BUILD}/
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

#updating Github Pages which host this documentation
find_package(Git REQUIRED)
if(Git_FOUND)
  add_custom_target(Deploy
                    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/deploy.sh ${SPHINX_BUILD}/old_html ${SPHINX_BUILD}/sphinx/html
                    COMMENT "Uploading documentation into Github Pages")
else()
  message(FATAL_ERROR "GIT cannot be found which is necessary for updating webpages")
endif()
