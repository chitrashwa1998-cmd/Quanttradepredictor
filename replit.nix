
{ pkgs }: {
  deps = [
    pkgs.lsof
    pkgs.nodejs_20
    pkgs.nodePackages.npm
    pkgs.gcc
    pkgs.glibc
    pkgs.glibcLocales
    pkgs.ocl-icd
    pkgs.opencl-headers
    pkgs.postgresql
    pkgs.stdenv.cc.cc.lib
    pkgs.libgcc
    pkgs.gcc-unwrapped.lib
    pkgs.libstdcxx5
    pkgs.zlib
  ];

  # Set library path for NumPy
  env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libgcc}/lib:${pkgs.gcc-unwrapped.lib}/lib";
  };
}
