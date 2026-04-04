DIRS = arma_einsum.hpp tests/

lint:
	 cpplint --quiet --recursive $(DIRS)

todo:
	grep -r "TODO(" $(DIRS)