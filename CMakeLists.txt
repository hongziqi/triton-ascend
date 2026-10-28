# This package is not added into a community Triton build.
# Scheme 1 plugin:
#   cmake -S standalone -B build/plugin -DLLVM_SYSPATH=$LLVM_SYSPATH
cmake_minimum_required(VERSION 3.20)
message(FATAL_ERROR
  "Build the Ascend plugin from standalone/ "
  "(links only libtriton; see standalone/README.md).")
