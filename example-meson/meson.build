project('test_einsum', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=3', 'cpp_std=c++20'],
)

cxx = meson.get_compiler('cpp')

# configure dependencies
project_dep = []

build_args = [
  '-DPROJECT_NAME="lib' + meson.project_name() + '"',
  '-DPROJECT_VERSION="' + meson.project_version() + '"',
]

arma_einsum_proj = subproject('arma-einsum', default_options: ['tests=false'])
arma_einsum_dep = arma_einsum_proj.get_variable('arma_einsum_dep')
project_dep += arma_einsum_dep

# source
srcs = ['main.cpp']

# Executable
test_exe = executable(
  meson.project_name(), 
  srcs,
  dependencies: project_dep,
  c_args : build_args,
  install: true
)
