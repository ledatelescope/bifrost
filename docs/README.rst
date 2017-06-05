Bifrost Documentation
=====================

Docker
------

To quickly build the docs using Docker, ensure that you have
built a Bifrost container as `ledatelescope/bifrost`.
Inside the `docs` folder, execute `./docker_build_docs.sh`,
which will create a container called `bifrost_docs`, then
run it, and have it complete the docs-building process for you,
outputting the entire html documentation inside `docs/html`.
