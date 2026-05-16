#pragma once

#include "nssolver/state.hpp"
#include "nssolver/types.hpp"

namespace nssolver {

struct GasModel {
    Real gamma {1.4};
    Real gas_constant {287.05};
    Real prandtl {0.72};
    Real mu_ref {1.716e-5};
    Real t_ref {273.15};
    Real sutherland {110.4};
    Real cv() const { return gas_constant / (gamma - 1.0); }
};

struct Freestream {
    Primitive primitive;
};

Real dynamic_viscosity(Real temperature, const GasModel& gas);

}  // namespace nssolver
