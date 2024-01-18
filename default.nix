{ pkgs ? import <nixpkgs> {} }:
with pkgs.python311Packages;
pkgs.mkShell {
  buildInputs = [
    (pkgs.python311.withPackages (ps: with ps; [
       numpy
       scipy
       pytorch
       opencv4
       torchvision
       yacs
       tqdm
    ]))
    # pkgs.python311Packages.virtualenv
    # pkgs.python311Packages.tornado
    # pkgs.python311Packages.aiohttp
  ];

  # shellHook = ''
  #   # Create a virtual environment and activate it
  #   python -m venv venv
  #   source venv/bin/activate
  #   pip install opencv3

  #   # You can add more setup steps if needed
  # '';
}

# { pkgs ? import <nixpkgs> {} }:

# let
#   pythonEnv = pkgs.python311.withPackages (ps: with ps; [
#     numpy
#     scipy
#     pytorch
#     torchvision
#     #opencv3
#     yacs
#     tqdm
#   ]);
# in
# pythonEnv
