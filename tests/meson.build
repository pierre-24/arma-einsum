# dependencies
tests_deps = [arma_einsum_dep]

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_dep')

tests_deps += gtest_dep

# create tests suite
tests = [
    'parser',
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