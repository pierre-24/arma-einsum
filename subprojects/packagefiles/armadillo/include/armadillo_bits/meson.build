config = configuration_data(
    {
        'ARMA_USE_WRAPPER': false,
        'ARMA_USE_LAPACK': get_option('use_lapack'),
        'ARMA_USE_BLAS': get_option('use_blas'),
        'ARMA_USE_ATLAS': get_option('use_atlas'),
        'ARMA_USE_ARPACK': get_option('use_arpack'),
        'ARMA_USE_SUPERLU': get_option('use_superlu'),
    }
)

configure_file(
    format: 'cmake@',
    input: 'config.hpp.cmake',
    output: 'config.hpp',
    configuration: config,
)