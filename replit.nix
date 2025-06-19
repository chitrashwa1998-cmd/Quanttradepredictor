{ pkgs }: {
  deps = [
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
    pkgs.glibc
    pkgs.gcc-unwrapped.lib
  ];
}