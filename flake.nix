{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python312.withPackages (ps:
              with ps; [
                graphviz

                # debugger
                pudb
              ]))
            gcc
            graphviz

            # linter and formatter
            ruff

            # spell and grammar check
            harper
          ];
          shellHook = ''
            export PYTHONPATH="$(realpath .):$(realpath ./interp_x86)"
            gcc -c -g -std=c99 runtime.c
          '';
          PYTHONBREAKPOINT="pudb.set_trace";
        };
      }
    );
}
