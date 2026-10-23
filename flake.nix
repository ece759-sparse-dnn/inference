{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          stdenv = pkgs.gcc15Stdenv;
          python = pkgs.python312;
        in
        {
          default = (pkgs.mkShell.override { inherit stdenv; }) {
            packages = [
              pkgs.cmake
              pkgs.ninja
              pkgs.clang-tools
              python
              pkgs.uv
            ];

            env = {
              UV_PYTHON = "${python}/bin/python3";
              UV_PYTHON_DOWNLOADS = "never";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                stdenv.cc.cc.lib
                pkgs.zlib
              ];
            };
          };
        }
      );
    };
}
