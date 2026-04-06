#include <benchmark/benchmark.h>

#include <armadillo>
#include <arma_einsum.hpp>

#define BENCH_FLOAT double

/* -- Trace */
static void BM_etrace(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        BENCH_FLOAT r = armaeinsum::einsum_mat_opt<BENCH_FLOAT>(armaeinsum::Greedy, "ii", A).at(0, 0);
        benchmark::DoNotOptimize(r);
    }
}

static void BM_atrace(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        BENCH_FLOAT r = arma::trace(A);
        benchmark::DoNotOptimize(r);
    }
}

BENCHMARK(BM_etrace)->Range(2<<5, 2<<10);
BENCHMARK(BM_atrace)->Range(2<<5, 2<<10);

/* -- daxpy */
static void BM_edaxpy(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));
    auto b = arma::randn<arma::Col<BENCH_FLOAT>>(state.range(0));

    for (auto _ : state) {
        arma::Mat<BENCH_FLOAT> r = armaeinsum::einsum_mat_opt<BENCH_FLOAT>(armaeinsum::Greedy, "ik,k->i", A, b);
    }
}

static void BM_adaxpy(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));
    auto b = arma::randn<arma::Col<BENCH_FLOAT>>(state.range(0));

    for (auto _ : state) {
        arma::Mat<BENCH_FLOAT> r = A * b;
    }
}

BENCHMARK(BM_edaxpy)->Range(2<<4, 2<<8);
BENCHMARK(BM_adaxpy)->Range(2<<4, 2<<8);

/* -- gemm */
static void BM_egemm(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));
    auto B = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        arma::Mat<BENCH_FLOAT> r = armaeinsum::einsum_mat_opt<BENCH_FLOAT>(armaeinsum::Greedy, "ik,kj->ij", A, B);
    }
}


static void BM_agemm(benchmark::State& state) {
    auto A = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));
    auto B = arma::randn<arma::Mat<BENCH_FLOAT>>(state.range(0), state.range(0));

    for (auto _ : state) {
        arma::Mat<BENCH_FLOAT> r = A * B;
    }
}

BENCHMARK(BM_egemm)->Range(2<<3, 2<<6);
BENCHMARK(BM_agemm)->Range(2<<3, 2<<6);

// Run the benchmark
BENCHMARK_MAIN();
