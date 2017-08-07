docker build -t bifrost_docs -f Dockerfile .
mkdir -p html
docker run --rm -i --net=none -v $(pwd)/html:/data bifrost_docs /bin/bash -c "cp -r ./* /data/"
