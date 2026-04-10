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

Conservative primitive_to_conservative(const Primitive& primitive, const GasModel& gas);
Primitive conservative_to_primitive(const Conservative& conservative, const GasModel& gas);
Real speed_of_sound(const Primitive& primitive, const GasModel& gas);
Real total_enthalpy(const Primitive& primitive, const GasModel& gas);
Real dynamic_viscosity(Real temperature, const GasModel& gas);
Real thermal_conductivity(Real temperature, const GasModel& gas);
Real eddy_viscosity(const Primitive& primitive, const GasModel& gas);
void update_primitives(FlowState& state, const GasModel& gas);
Primitive sample_primitive(const FlowState& state, Index i);
Conservative sample_conservative(const FlowState& state, Index i);

}  // namespace nssolver
