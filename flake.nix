{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nixrik = {
      url = "gitlab:erikmannerfelt/nixrik";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nixrik, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        my-python = nixrik.packages.${system}.python_from_requirements {python_packages = pkgs.python311Packages;} ./requirements.txt;

        packages = builtins.listToAttrs (map (pkg: { name = pkg.pname; value = pkg; }) ( with pkgs; [
          #(import ./python.nix { inherit pkgs; })
          pre-commit
          zsh
          google-cloud-sdk
          gifsicle
          ffmpeg
        ])) // {python=my-python;};

      in
      {
        inherit packages;
        defaultPackage = packages.python;

        devShell = pkgs.mkShell {
          name = "GlobalSurgeDetection";
          buildInputs = pkgs.lib.attrValues packages;
          shellHook = ''

            zsh
          '';
        };
      }

    );
}
