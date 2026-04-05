DIRS = arma_einsum.hpp tests/ benchmarks/

lint:
	 cpplint --quiet --recursive $(DIRS)

todo:
	grep -r "TODO(" $(DIRS)