Example of project using `arma-einsum` via Meson.

Use [`arma-einsum.wrap`](subprojects/arma-einsum.wrap) in your `subproject` directory, and then:

```meson
arma_einsum_proj = subproject('arma-einsum', default_options: ['tests=false'])
arma_einsum_dep = arma_einsum_proj.get_variable('arma_einsum_dep')

# `project_dep` contains your project dependencies
project_dep += arma_einsum_dep
```

See [`meson.build`](meson.build) for the full example.

You can build the example using:

```bash
# use Meson
meson setup _build
meson compile _build

# run example, should print "55"
./_build/test_einsum
```