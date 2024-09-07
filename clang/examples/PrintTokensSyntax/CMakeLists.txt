add_llvm_library(PrintTokensSyntax MODULE PrintTokensSyntax.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(PrintTokensSyntax PRIVATE
    clangAST
    clangBasic
    clangParse
    clangLex
    )
endif()
