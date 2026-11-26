# ONERA M6 Validation Case

The ONERA M6 wing in inviscid transonic flow, on SU2's public tutorial mesh and
free stream, against Schmitt and Charpin's experiment:

- Tutorial: https://su2code.github.io/tutorials/Inviscid_ONERAM6/
- Source mesh: `public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2`
- Source config: `public-data/onera_m6/inv_ONERAM6.cfg`
- Experiment: V. Schmitt and F. Charpin, "Pressure distributions on the ONERA-M6-wing at
  transonic Mach numbers", AGARD AR-138 (1979), test 2308; the surface pressure at seven span
  stations from NASA's NPARC archive, `public-data/onera_m6/ONERAb114.tec`
  (https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html)

`docs/VALIDATION.md` section 10 summarizes the case among the other verification runs.

## Download and conversion

`scripts/download_onera_m6_su2_case.sh` fetches SU2's configuration
(`su2code/Tutorials`), extracts the mesh from that repository's archive and
downloads the experiment's sections, into `public-data/onera_m6/`. The run
script calls it, and preprocesses the mesh afresh each time (a second or so):

```bash
.helpers/bin/nssolver_preprocess_op2_helper onera_m6 meshes-op2/onera_m6.h5 public-data/onera_m6/mesh_ONERAM6_inv_ffd.su2
```

The imported mesh has:

- nodes: `108396`
- tetrahedra in the SU2 mesh: `582752`
- dual edges: `710525`
- boundary faces (triangles): `38756`

SU2's markers, in the mesh's order, become boundary groups 1-7: `LOWER_SIDE`,
`TIP`, `UPPER_SIDE` (walls, groups 1-3), `XNORMAL_FACES`, `ZNORMAL_FACES`,
`YNORMAL_FACE` (far field), `SYMMETRY_FACE` (the root's symmetry plane). The
wing's semi-span b is 1.1963 m, where the rounded tip cap starts.

## Free stream

From the SU2 tutorial's inputs (M 0.8395, 3.06 degrees, 101325 Pa, 288.15 K,
$\gamma$ 1.4, R 287.05 J/(kg K)):

- `rho_inf = 1.2250122659906946 kg/m^3`
- `u_inf = 285.26805348028574 m/s`
- `v_inf = 0.0 m/s`
- `w_inf = 15.249834230540356 m/s`

## Running the case

```bash
BIN=build/nssolver_op2_cuda scripts/run_onera_m6_validation.sh --solve-config configs/solve.json
```

`configs/onera_m6.cfg`: inviscid, second order with the van Albada limiter,
the implicit scheme from CFL 10 growing by 1.3 a step to 1e4, 150 iterations.
The script then runs `nssolver_onera_m6_validation_helper` on the solution: the
pressure force on the wall groups, its coefficients on the planform area (the
walls' upward-facing projected area), the wall's face $C_p$, and the wall cut at
the experiment's stations y/b = 0.20, 0.44, 0.65, 0.80, 0.90, 0.96 and 0.99,
which `scripts/onera_m6_sections.py` compares with the experiment.

Outputs in `outputs-op2/`:

- `onera_m6_solution.h5`, `onera_m6_solution.residual.csv`
- `onera_m6_summary.json`: reference area, force and coefficients
- `onera_m6_surface_cp.csv`: each wall face's centroid and $C_p$
- `onera_m6_sections.csv`: the stations' cuts (station, group, x, z, $C_p$)
- `onera_m6_sections.png`: computed against experimental sections, where matplotlib is installed
- `onera_m6_solution.vtk`

## Results

The density residual falls ten orders, to round-off ($10^{-13}$), in 150
iterations: 2 minutes on a laptop GPU (RTX 4050), 3300 linear iterations, no
rejected updates. A first-order run converges in 30 iterations.

- reference area `0.758799`
- `CL = 0.287754`, `CD = 0.0130338` (pressure only, inviscid)

The $C_p$ RMS difference from the experiment, at the experiment's points:

| y/b | 0.20 | 0.44 | 0.65 | 0.80 | 0.90 | 0.96 | 0.99 |
|---|---|---|---|---|---|---|---|
| upper | 0.105 | 0.100 | 0.076 | 0.076 | 0.114 | 0.069 | 0.115 |
| lower | 0.054 | 0.040 | 0.026 | 0.028 | 0.033 | 0.025 | 0.033 |

The lower surface, the leading-edge suction peaks and the lambda pattern (two
upper-surface shocks at 0.44 and 0.65, merged from 0.80 out) match. The
upper-surface differences are the shocks' positions: the computed shocks sit a
few percent of the chord aft of the experiment's and are sharper, as expected
without a boundary layer to thicken their foot. At y/b 0.99 the experiment's aft upper surface
carries the tip vortex's suction, which the computation lacks. The experiment's
stated accuracy is $\pm 0.02$ in $C_p$.
