# Flowlab

Real-time incompressible fluid and smoke simulation that runs entirely in the
browser on WebGPU, with a CPU reference implementation used to validate every
number the GPU produces.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ca814ff9-86a8-430b-b482-f598fd75ddcc" alt="Flowlab demo — smoke curling off a vortex street" width="720">
</p>

<p align="center">
  <a href="https://aleksandrrogachev94.github.io/flowlab/"><strong>Live demo →</strong></a>
</p>

## What this is

A solver for the incompressible Navier–Stokes equations on a staggered MAC
grid, following Robert Bridson's
[_Fluid Simulation for Computer Graphics_](https://www.cs.ubc.ca/~rbridson/fluidsimulation/)
for the projection method and boundary handling, with MacCormack advection
(Selle et al., 2008) in place of plain semi-Lagrangian for a sharper,
less dissipative result. Built to actually understand the numerics rather
than wrap a library. It exists in two forms that are meant to agree with
each other:

- **A CPU reference** in plain TypeScript, float64, easy to read and cheap to
  test headlessly.
- **A GPU implementation** in WGSL compute shaders, float32, fast enough to
  run interactively at real resolutions.

Every kernel — advection, pressure projection, buoyancy, vorticity — has a
CPU and a GPU version that are diffed against each other in tests, so a
plausible-looking but wrong GPU result has nowhere to hide.

## Scenes

- **Vortex street** — flow past a cylinder, the classic Kármán case
- **Thermal plume** — buoyant smoke rising off one or two heat sources
- **Turbulence grid** — decaying turbulence behind a grid of obstacles
- **Wing section** — flow over an airfoil at adjustable angle of attack
- **Wall jet**, **vortex cluster**, **dipole** — canonical test flows for
  checking the solver against known behavior

Each scene exposes live controls for resolution, solver, dye pattern, and
which field to visualize (dye, pressure, vorticity, velocity vectors).

## Running it locally

```bash
npm install
npm run dev
```

Requires a browser with WebGPU (Chrome/Edge, or recent Firefox/Safari).
Without WebGPU support the CPU engine still runs, just without the
interactive frame rate.

```bash
npm run build       # type-check + production build
npm test            # CPU kernel tests, ~1s, no browser
npm run test:gpu    # GPU kernels vs. their CPU twins, headless Chrome
npm run bench       # phase-by-phase timing, CPU vs GPU
```

## How it works

Each step is the standard projection method: advect velocity and dye,
compute divergence, solve for pressure, subtract the pressure gradient. Five
solver/ordering combinations exist side by side (CPU: SOR, red-black SOR,
multigrid; GPU: red-black SOR, multigrid) specifically so a wrong-looking
result on the GPU can be isolated to precision, ordering, or plumbing rather
than guessed at.

The full set of engineering rules behind that — why kernels never mutate
in place, why the pressure solver is the one deliberate exception, why
`Simulation.step()` is `async` — is written up in
[ARCHITECTURE.md](ARCHITECTURE.md). The numerical background and the
project roadmap are in [PLAN.md](PLAN.md), and GPU-specific gotchas
(precision, workgroup sizing, secure-context requirements) are in
[docs/WEBGPU.md](docs/WEBGPU.md).

## Stack

TypeScript, Vite, WGSL. No UI framework, no simulation dependencies —
the whole thing is `TypeScript`, `Vite`, and `@webgpu/types`.

## License

[MIT](LICENSE)
