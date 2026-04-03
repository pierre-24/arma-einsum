project(
    'armadillo', 'cpp',
    version : '14.2.x',
)

subdir('include/armadillo_bits')
includes = include_directories('include', is_system: not meson.is_subproject())

armadillo_dep = declare_dependency(
    include_directories: includes
)