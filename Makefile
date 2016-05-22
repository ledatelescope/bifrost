
all:
	$(MAKE) -C ./src all
test:
	$(MAKE) -C ./src test
install:
	cp lib/*bifrost*  /usr/local/lib/
	cp -r src/bifrost /usr/local/include/
clean:
	$(MAKE) -C ./src clean
