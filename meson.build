project(
    'arma-einsum', 'cpp',
    version : '0.2.0',
    default_options : ['warning_level=3', 'cpp_std=c++20']
)

cxx = meson.get_compiler('cpp')

build_args = [
    '-DPROJECT_NAME="' + meson.project_name() + '"',
    '-DPROJECT_VERSION="' + meson.project_version() + '"',
]

date_program = find_program('date', required: false)
if date_program.found()
    build_args += '-DPROJECT_BUILD_DATE="' + run_command(date_program).stdout().replace('\n', '') + '"'
endif

git_program = find_program('git', required: false)
if git_program.found()
    build_args += '-DPROJECT_BUILD_COMMIT="' + run_command(git_program, 'rev-parse', 'HEAD').stdout().substring(0, 7) + '"'
endif

includes = include_directories('.')

# -- DEPENDENCIES
lib_deps = []

# add armadillo
armadillo_proj = subproject('armadillo', default_options: ['use_lapack=true', 'use_blas=true'])
armadillo_dep = armadillo_proj.get_variable('armadillo_dep')
lib_deps += armadillo_dep

# add OpenMP
if cxx.get_id() == 'clang'
    lib_deps +=  declare_dependency(
        compile_args: '-fopenmp',
        link_args: '-fopenmp',
    )
else
    lib_deps += dependency('openmp', language: 'cpp')
endif

arma_einsum_dep = declare_dependency(
    include_directories: includes,
    dependencies: lib_deps
)

if get_option('tests')
    subdir('tests')
    subdir('benchmarks')
endif