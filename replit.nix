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
  ];
}