#include <benchmark/benchmark.h>

#include <armadillo>
#include <arma_einsum.hpp>

#define BENCH_FLOAT double

/* -- Trace */
static void BM_etrace(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        BENCH_FLOAT r = armaeinsum::einsum_mat<BENCH_FLOAT>("ii->", A).at(0, 0);
        benchmark::DoNotOptimize(r);
    }
}

BENCHMARK(BM_etrace)->Range(2<<5, 2<<10);

static void BM_atrace(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        BENCH_FLOAT r = arma::trace(A);
        benchmark::DoNotOptimize(r);
    }
}

BENCHMARK(BM_atrace)->Range(2<<5, 2<<10);

/* -- gemm */
static void BM_egemm(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));
    auto B = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        arma::Mat<BENCH_FLOAT> r = armaeinsum::einsum_mat<BENCH_FLOAT>("ik,kj->ij", A, B);
    }
}

BENCHMARK(BM_egemm)->Range(2<<3, 2<<6);

static void BM_agemm(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));
    auto B = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        arma::Mat<BENCH_FLOAT> r = A * B;
    }
}

BENCHMARK(BM_agemm)->Range(2<<3, 2<<6);

// Run the benchmark
BENCHMARK_MAIN();
