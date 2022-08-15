{
  description =
    "A stream processing framework for high-throughput applications.";

  inputs.ctypesgen = {
    url = "github:ctypesgen/ctypesgen/ctypesgen-1.0.2";
    flake = false;
  };

  inputs.pre-commit-hooks = {
    url = "github:cachix/pre-commit-hooks.nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs@{ self, nixpkgs, pre-commit-hooks, ... }:
    let
      inherit (nixpkgs) lib;

      # Parse the version info in the AC_INIT declaration.
      acVersion = lib.head
        (builtins.match "AC_INIT\\(\\[bifrost], *\\[([.0-9]+)].*" (lib.head
          (lib.filter (lib.strings.hasPrefix "AC_INIT")
            (lib.splitString "\n" (lib.readFile ./configure.ac)))));

      # Add a git hash if available; but if repo isn't clean then flake won’t
      # provide shortRev and version ends in ".dev".
      version = "${acVersion}.dev"
        + lib.optionalString (self ? shortRev) "+g${self.shortRev}";

      compilerName = stdenv:
        lib.replaceStrings [ "-wrapper" ] [ "" ] stdenv.cc.pname;

      # Can inspect the cuda version to guess at what architectures would be
      # most useful. Take care not to instatiate the cuda package though, which
      # would happen if you start inspecting header files or trying to run nvcc.

      defaultGpuArchs = cudatoolkit:
        if lib.hasPrefix "11." cudatoolkit.version then [
          "80"
          "86"
        ] else [
          "70"
          "75"
        ];

      # At time of writing (2022-03-24):
      # PACKAGE          VERSION ARCHS
      # cudatoolkit      │       │
      #  ≡ ~_10          │       │
      #  ≡ ~_10_2        10.2.89 30 32 35 37 50 52 53 60 61 62 70 72 75
      # cudatoolkit_11   │       │  (deprecated)
      #  ≡ ~_11_4        11.4.2  │    (35 37 50)52 53 60 61 62 70 72 75 80 86 87
      # cudatoolkit_11_5 11.5.0  │    (35 37 50)52 53 60 61 62 70 72 75 80 86 87
      #                  │       │*Experimented w/using all supported archs, but
      #                  │       │ had to eliminate 87 because not in cufft lib.

      bifrost = { stdenv, ctags, ncurses, file, enableDebug ? false
        , enablePython ? true, python3, enableCuda ? false, cudatoolkit
        , util-linuxMinimal, gpuArchs ? defaultGpuArchs cudatoolkit }:
        stdenv.mkDerivation {
          name = lib.optionalString (!enablePython) "lib" + "bifrost-"
            + compilerName stdenv + lib.versions.majorMinor stdenv.cc.version
            + lib.optionalString enablePython
            "-py${lib.versions.majorMinor python3.version}"
            + lib.optionalString enableCuda
            "-cuda${lib.versions.majorMinor cudatoolkit.version}"
            + lib.optionalString enableDebug "-debug" + "-${version}";
          inherit version;
          src = ./.;
          buildInputs = [ stdenv ctags ncurses ] ++ lib.optionals enablePython [
            python3
            python3.pkgs.ctypesgen
            python3.pkgs.setuptools
            python3.pkgs.pip
            python3.pkgs.wheel
          ] ++ lib.optionals enableCuda [ cudatoolkit util-linuxMinimal ];
          propagatedBuildInputs = lib.optionals enablePython [
            python3.pkgs.contextlib2
            python3.pkgs.graphviz
            python3.pkgs.matplotlib
            python3.pkgs.numpy
            python3.pkgs.pint
            python3.pkgs.scipy
            python3.pkgs.simplejson
          ];
          patchPhase =
            # libtool wants file command, and refers to it in /usr/bin
            ''
              sed -i 's:/usr/bin/file:${file}/bin/file:' configure
            '' +
            # Use pinned ctypesgen, not one from pypi.
            ''
              sed -i 's/ctypesgen==1.0.2/ctypesgen/' python/setup.py
            '' +
            # Mimic the process of buildPythonPackage, which explicitly
            # creates wheel, then installs with pip.
            ''
              sed -i -e "s:build @PYBUILDFLAGS@:bdist_wheel:" \
                  -e "s:@PYINSTALLFLAGS@ .:${
                    lib.concatStringsSep " " [
                      "--prefix=${placeholder "out"}"
                      "--no-index"
                      "--no-warn-script-location"
                      "--no-cache"
                    ]
                  } dist/*.whl:" \
                  python/Makefile.in
            '';
          # Had difficulty specifying this with configureFlags, because it
          # wants to quote the args and that fails with spaces in gpuArchs.
          configurePhase = lib.concatStringsSep " "
            ([ "./configure" "--disable-static" ''--prefix="$out"'' ]
              ++ lib.optionals enableDebug [ "--enable-debug" ]
              ++ lib.optionals enableCuda [
                "--with-cuda-home=${cudatoolkit}"
                ''--with-gpu-archs="${lib.concatStringsSep " " gpuArchs}"''
                "--with-nvcc-flags='-Wno-deprecated-gpu-targets'"
                "LDFLAGS=-L${cudatoolkit}/lib/stubs"
              ]);
          preBuild = lib.optionalString enablePython ''
            make -C python bifrost/libbifrost_generated.py
            sed -e "s:^add_library_search_dirs(\[:&'$out/lib':" \
                -e 's:name_formats = \["%s":&,"lib%s","lib%s.so":' \
                -i python/bifrost/libbifrost_generated.py
          '';
          # This can be a helpful addition to above sed; prints each path
          # tried when loading library:
          # -e "s:return self\.Lookup(path):print(path); &:" \
          makeFlags =
            lib.optionals enableCuda [ "CUDA_LIBDIR64=$(CUDA_HOME)/lib" ];
          preInstall = ''
            mkdir -p "$out/lib"
          '';
        };

      ctypesgen =
        { buildPythonPackage, setuptools-scm, toml, glibc, stdenv, gcc }:
        buildPythonPackage rec {
          pname = "ctypesgen";
          # Setup tools won’t be able to run git describe to generate the
          # version, but we can include the shortRev.
          version = "1.0.2.dev+g${inputs.ctypesgen.shortRev}";
          SETUPTOOLS_SCM_PRETEND_VERSION = version;
          src = inputs.ctypesgen;
          buildInputs = [ setuptools-scm toml ];
          postPatch =
            # Version detection in the absence of ‘git describe’ is broken,
            # even with an explicit VERSION file.
            ''
              sed -e 's/\(VERSION = \).*$/\1"${pname}-${version}"/' \
                  -e 's/\(VERSION_NUMBER = \).*$/\1"${version}"/' \
                  -i ctypesgen/version.py
            '' +
            # Test suite invokes ‘run.py’, replace that with actual script.
            ''
              sed -e "s:\(script = \).*:\1'${
                placeholder "out"
              }/bin/ctypesgen':" \
                  -e "s:run\.py:ctypesgen:" \
                  -i ctypesgen/test/testsuite.py
            '' +
            # At runtime, ctypesgen invokes ‘gcc -E’. It won’t be available in
            # the darwin stdenv so let's explicitly patch full path to gcc in
            # nix store, making gcc a true prerequisite, which it is. There
            # are also runs of gcc specified in test suite.
            ''
              sed -i 's:gcc -E:${gcc}/bin/gcc -E:' ctypesgen/options.py
            '' +
            # Some tests explicitly load ‘libm’ and ‘libc’. They won’t be
            # found on NixOS unless we patch in the ‘glibc’ path.
            lib.optionalString stdenv.isLinux ''
              sed -e 's:libm.so.6:${glibc}/lib/&:' \
                  -e 's:libc.so.6:${glibc}/lib/&:' \
                  -i ctypesgen/test/testsuite.py
            '';
          checkPhase = "python ctypesgen/test/testsuite.py";
        };

      pyOverlay = self: _: {
        ctypesgen = self.callPackage ctypesgen { };
        bifrost = self.toPythonModule (self.callPackage bifrost {
          enablePython = true;
          python3 = self.python;
        });
      };

      bifrost-doc =
        { stdenv, python3, ctags, doxygen, docDir ? "/share/doc/bifrost" }:
        stdenv.mkDerivation {
          name = "bifrost-doc-${version}";
          inherit version;
          src = ./.;
          buildInputs = [
            ctags
            doxygen
            python3
            python3.pkgs.bifrost
            python3.pkgs.sphinx
            python3.pkgs.breathe
          ];
          buildPhase = ''
            make doc
            make -C docs html
            cd docs/build/html
            mv _static static
            mv _sources sources
            find . -type f -exec sed -i \
              -e '/\(href\|src\)=\"\(\.\.\/\)\?_static/ s/_static/static/' \
              -e '/\(href\|src\)=\"\(\.\.\/\)\?_modules/ s/_modules/modules/' \
              -e '/\(href\|src\)=\"\(\.\.\/\)\?_sources/ s/_sources/sources/' \
              -e '/\$\.ajax\(.*\)_sources/ s/_sources/sources/' \
              {} \;
            cd ../../..
          '';
          installPhase = ''
            mkdir -p "$out${docDir}"
            cp -r docs/build/html "$out${docDir}"
          '';
        };

      # Enable pre-configured packages for these systems.
      eachSystem = do:
        lib.genAttrs [ "x86_64-linux" "x86_64-darwin" ] (system:
          do (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = lib.attrValues self.overlays;
          }));

      # Which python3 packages should be modified by the overlay?
      isPython = name: builtins.match "python3[0-9]*" name != null;
      pythonAttrs = lib.filterAttrs (name: _: isPython name);

    in {
      overlays.default = final: prev:
        {
          bifrost = final.callPackage bifrost { };
          bifrost-doc = final.callPackage bifrost-doc { };
          github_stats = final.writeShellScriptBin "github_stats" ''
            ${final.python3.withPackages (p: [ p.PyGithub ])}/bin/python \
              ${tools/github_stats.py} "$@"
          '';
        }
        # Apply the python overlay to every python package set we find.
        // lib.mapAttrs (_: py: py.override { packageOverrides = pyOverlay; })
        (pythonAttrs prev);

      packages = eachSystem (pkgs:
        let
          shortenPy = lib.replaceStrings [ "thon" ] [ "" ];

          # Which cuda versions should be target by the packages? Let's just do
          # the default 10 and 11. It's easy to generate other point releases
          # from the overlay. (Versions prior to 10 are not supported anymore by
          # nixpkgs.)
          isCuda = name: builtins.match "cudaPackages(_1[01])" name != null;
          shortenCuda = lib.replaceStrings [ "Packages" "_" ] [ "" "" ];
          cudaAttrs = lib.filterAttrs (name: pkg:
            isCuda name && lib.elem pkgs.system pkg.cudatoolkit.meta.platforms)
            pkgs;

          # Which C++ compilers can we build with? How to name them?
          eachCxx = f:
            lib.concatMap f (with pkgs; [
              stdenv
              gcc8Stdenv
              gcc9Stdenv
              gcc10Stdenv
              gcc11Stdenv
              clang6Stdenv
              clang7Stdenv
              clang8Stdenv
              clang9Stdenv
              clang10Stdenv
            ]);
          cxxName = stdenv:
            lib.optionalString (stdenv != pkgs.stdenv)
            ("-" + compilerName stdenv + lib.versions.major stdenv.cc.version);

          eachBool = f: lib.concatMap f [ true false ];
          eachCuda = f: lib.concatMap f ([ null ] ++ lib.attrNames cudaAttrs);
          eachConfig = f:
            eachBool (enableDebug:
              eachCuda (cuda:
                eachCxx (stdenv:
                  f (cxxName stdenv
                    + lib.optionalString (cuda != null) "-${shortenCuda cuda}"
                    + lib.optionalString enableDebug "-debug") {
                      inherit stdenv enableDebug;
                      enableCuda = cuda != null;
                      cudatoolkit = pkgs.${cuda}.cudatoolkit;
                    })));

          # Runnable ctypesgen per python. Though it's just the executable we
          # need, it's possible something about ctypes library could change
          # between releases.
          cgens = lib.mapAttrs' (name: py: {
            name = "ctypesgen-${shortenPy name}";
            value = py.pkgs.ctypesgen;
          }) (pythonAttrs pkgs);

          # The whole set of bifrost packages, with or without python (and each
          # python version), and for each configuration.
          bfs = lib.listToAttrs (eachConfig (suffix: config:
            [{
              name = "libbifrost${suffix}";
              value =
                pkgs.bifrost.override (config // { enablePython = false; });
            }] ++ lib.mapAttrsToList (name: py: {
              name = "bifrost-${shortenPy name}${suffix}";
              value = py.pkgs.bifrost.override config;
            }) (pythonAttrs pkgs)));

          # Now generate pythons with bifrost packaged.
          pys = lib.listToAttrs (eachConfig (suffix: config:
            lib.mapAttrsToList (name: py: {
              name = "${name}-bifrost${suffix}";
              value = (py.withPackages
                (p: [ (p.bifrost.override config) ])).override {
                  makeWrapperArgs = lib.optionals config.enableCuda
                    [ "--set LD_PRELOAD /usr/lib/x86_64-linux-gnu/libcuda.so" ];
                };
            }) (pythonAttrs pkgs)));

        in { inherit (pkgs) bifrost-doc github_stats; } // cgens // bfs // pys);

      devShells = eachSystem (pkgs: {
        default = let
          pre-commit = pre-commit-hooks.lib.${pkgs.system}.run {
            src = ./.;
            hooks.nixfmt.enable = true;
            hooks.nix-linter.enable = true;
            hooks.yamllint.enable = true;
            hooks.yamllint.excludes = [ ".github/workflows/main.yml" ];
          };

        in pkgs.mkShellNoCC {
          inherit (pre-commit) shellHook;

          # Tempting to include bifrost-doc.buildInputs here, but that requires
          # bifrost to already be built.
          buildInputs = pkgs.bifrost.buildInputs
            ++ pkgs.bifrost.propagatedBuildInputs ++ [
              pkgs.black
              pkgs.ctags
              pkgs.doxygen
              pkgs.nixfmt
              pkgs.nix-linter
              pkgs.python3.pkgs.breathe
              pkgs.python3.pkgs.sphinx
              pkgs.yamllint
            ];
        };
      });
    };
}
