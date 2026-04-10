#pragma once

#include <array>

#include "nssolver/mesh.hpp"
#include "nssolver/physics.hpp"

namespace nssolver {

using FluxArray = std::array<Real, 6>;

FluxArray physical_flux(const Primitive& primitive, const Conservative& conservative, const Vec3& normal, const GasModel& gas);
FluxArray hllc_flux(const Primitive& left_p,
                    const Conservative& left_q,
                    const Primitive& right_p,
                    const Conservative& right_q,
                    const Vec3& area_vector,
                    const GasModel& gas);
FluxArray thin_layer_viscous_flux(const Primitive& left_p,
                                  const Primitive& right_p,
                                  const Vec3& delta_r,
                                  const Vec3& area_vector,
                                  const GasModel& gas);
Primitive make_boundary_ghost_state(const Primitive& interior,
                                    const Primitive& freestream,
                                    const Vec3& unit_normal,
                                    BoundaryType boundary_type);

}  // namespace nssolver
