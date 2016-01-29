MAKEFILES:
The central makefile which is to be used to compile all the versions of ChASE is in the file "Makefile". It contains the definitions of the source, header and object files, their locations and the rules how to make the different versions. The idea is that the makefile itself does not have to be modified whenever ChASE is installed on different machines. The architecture (and system) specific environment variables are contained in separate .conf files, each file for a different architecture. The variables there are to be changed and adapted to the system that ChASE is supposed to run on.
For better organization of the makefile, it is recomended that the files in each version are arranged according to a scheme:
On the highest level, just under the version number folder, the diferent implementations (Ele, CUDA, Phi) od ChASE should be in a folder, which is named just as the architecture, and without the version number. Inside this folder, the different files should be divided according to their types in differtnt folrers (include, src, obj). No main fles, executables, drivers or interfaces should be present here, as this will be the core of the ChASE library.
The drivers, executables and inrerfaces are to be placed in "example drivers" folder, just under the version number level. Then they should also be divided in folders according to their architecture and purpose. The makefile is not strict about them so far, as it is intended to compile ChASE up to a library level format. The compilation of drivers, interfaces and executables can be made by the makefile by adding additional rules for them.

PROGRESS ON THE MAKEFILES:
The makefile is not finished yet!
My idea was to make it as generic as possible. This however, did not work in practice, and I tried to make some ugly fixes in the phi-rule (which makes the object files for the xeon-phi version). It works, in the sense of producing the desired object files, but it is not putting them in the "../phiChASE/obj".
If this is resolved, then the same approach (although ugly) may be used to write the other rules.

Regarding the Elemental version of the Makefile, not much was done there, as the 0.4 version of EleChASE was not finished up to the point when this document was written.

WHAT NEEDS TO BE DONE:

Besides fixing the current rules, the rules in the makefile should look like this:

1. Rules that make .o files (compile-only rules). These object files are then to be used by other rules, namely:
2. Rules that build a library: Separate libraries should be built out of the different versions (Ele, Phi, CUDA), out of the already compiled .o files.
3. Rules that create executables: As mentioned, specific drivers, examples, and interfaces may be provided with ChASE. Rules for them should be added, which will also take the already compiled .o files. Or include the libraries.
