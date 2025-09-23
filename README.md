# Canopy

### Analytical methods:
1. [Fast multipole info](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/2012_fmm_encyclopedia.pdf)

2. [Lecture 2](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Lectures/lecture02.pdf)

3. [FMM for Vortical Flows](https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content)

4. [1987_greengard_dissertation](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/1987_greengard_dissertation.pdf)

5. [CSCAMM Lecture](https://home.cscamm.umd.edu/programs/fam04/dg_lecture6.pdf)

### Black Box Methods:
6. [The black-box fast multipole method](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf)

### Workflow for upwards sweep:
1. Get raw particle x/y/z and scalar data in AoSoA format.
2. Iterate over particles. Activate the appropriate cells in the mesh based on particle positions.
3. Fill leaf layer:
    1. Get AoSoA of particles + scalar data that reside within the cell.
    2. Convert particles into "num_M" multipoles.
    3. Store multipoles in _M, indexed by unique, per-process, contiguous cell ids, called ccell_id. Cell with id "ccell_id" has mutlipoles at _M[num_M*ccell_id, (num_M+1) * ccell_id).
    4. Store ccell_id and the cell x/y/z center in the mesh at each cell.
4. Fill non-leaf layers:
    1. Get cell x/y/z centers from the mesh.
    2. Determine the parent cell based on x/y/z center.
    3. Send multipole coefficients and cell center to the rank which owns the parent cell.
    4. Per cell, parent rank uses M2M operation to translate and add multipole coefficients.
    5. Stores these coefficients in its _M array with the same indexing scheme.
