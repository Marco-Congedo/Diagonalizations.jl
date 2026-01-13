#   This script is not part of the Diagonalizations.jl package.
#   It allows to build the package
#   and its documentation locally from the source code,
#   without actually installing the package.
#   It is used for developing purposes using the Julia
#   `Revise` package (that you need to have installed on your PC,
#   together with the `Documenter` package for building the documentation).
#   You won't need this script for using the package.
#
#   DIRECTIONS:
#   1) If you have installed the Diagonalizations.jl package
#      from github or Julia registry, uninstall it.
#   2) Change the `projectDir` path here below to the path
#           where the "diagonalization" folder is located on your PC.
#   3) Under Linux, replace all '\\' with `/`
#   4) Put the cursor in this unit and hit SHIFT+CTRL+ENTER
#
#   Nota Bene: all you need for building the package is actually
#   the 'push' line and the 'using' line.
#   You can safely delete the rest once
#   you have identified the 'srcDir' to be used in the push command.

begin
    # change the 'projectDir' path to the folder where your project is
    projectDir = homedir()*"\\Documents\\Code\\julia\\Diagonalizations"
    push!(LOAD_PATH, projectDir*"\\src\\")
    using Revise, Diagonalizations
end

