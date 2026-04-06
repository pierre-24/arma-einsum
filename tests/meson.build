# dependencies
tests_deps = [arma_einsum_dep]

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_dep')

tests_deps += gtest_dep

# linear algebra backend
foreach pkg: get_option('la_packages')
    tests_deps += dependency(pkg, required: true)
endforeach

foreach lib: get_option('la_libraries')
    tests_deps += cc.find_library(lib, required : true)
endforeach

# create tests suite
tests = [
    'parser',
    'iterator',
    'contraction',
    'einsum',
]

foreach t: tests
    test_exec = executable(
        'tests_suite_' + t,
        sources: [files('tests_' + t + '.cpp'), 'main.cpp'],
        include_directories : include_directories('.'),
        dependencies: tests_deps,
        install: false,
        build_by_default: false,
        cpp_args : build_args,
    )

    test('tests_suite_' + t,
         test_exec,
         workdir : meson.current_source_dir(),
         suite: 'lib_code',
    )
endforeach