sphinx-apidoc -o source -d 5 --force ../python/bifrost/
rm source/modules.rst
sed -i '1s/.*/Python Reference/' source/bifrost.rst
sed -i '2s/.*/================/' source/bifrost.rst
sed -i '1s/.*/Block Library Reference/' source/bifrost.blocks.rst
sed -i '2s/.*/=======================/' source/bifrost.blocks.rst
