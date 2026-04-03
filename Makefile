DIRS = . tests/

lint:
	 cpplint --quiet --recursive $(DIRS)

todo:
	grep -r "TODO(" $(DIRS)