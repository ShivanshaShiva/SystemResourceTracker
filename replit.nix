{pkgs}: {
  deps = [
    pkgs.python311Packages.google-cloud-core
    pkgs.python311Packages.google-cloud-workflows
    pkgs.deepin.dwayland
    pkgs.sbclPackages.qtools-ui-spellchecked-text-edit
    pkgs.python311Packages.qiskit-terra
    pkgs.python311Packages.qiskit-aer
    pkgs.python312Packages.qiskit-terra
  ];
}
