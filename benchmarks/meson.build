# dependencies
bench_deps = [arma_einsum_dep]

# gbench
gbench_proj = subproject('google-benchmark')
gbench_dep = gbench_proj.get_variable('google_benchmark_dep')
bench_deps += gbench_dep

# linear algebra backend
foreach pkg: get_option('la_packages')
    bench_deps += dependency(pkg, required: true)
endforeach

foreach lib: get_option('la_libraries')
    bench_deps += cc.find_library(lib, required : true)
endforeach

bench_exec = executable(
    'arma_einsum_benchmark',
    sources: ['bench.cpp'],
    include_directories : include_directories('.'),
    dependencies: bench_deps,
    install: false,
    build_by_default: false,
    cpp_args : build_args,
)

benchmark('bench_arma_einsum',
     bench_exec,
     workdir : meson.current_source_dir(),
     suite: 'lib_code',
)