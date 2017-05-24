docker build -t bifrost_docs -f Dockerfile .
mkdir -p out
docker run --rm -i --net=none -v $(pwd)/out:/data /bin/bash -c "cp -r ./* /data/"
