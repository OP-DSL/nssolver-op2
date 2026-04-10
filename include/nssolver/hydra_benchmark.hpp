#pragma once

#include <string>

#include "nssolver/physics.hpp"

namespace nssolver {

struct HydraInflowProfile {
    std::size_t rows {};
    Real mean_radius {};
    Real total_temperature {};
    Real total_pressure {};
    Real whirl_angle {};
    Real pitch_angle {};
    Real nu_tilde {};
};

struct HydraOutflowProfile {
    std::size_t rows {};
    Real mean_radius {};
    Real static_pressure {};
};

struct HydraBenchmarkConditions {
    Primitive primitive;
    Real mach {};
    Real inlet_total_temperature {};
    Real inlet_total_pressure {};
    Real original_exit_static_pressure {};
};

HydraInflowProfile read_hydra_inflow_profile(const std::string& path);
HydraOutflowProfile read_hydra_outflow_profile(const std::string& path);
HydraBenchmarkConditions make_hydra_benchmark_conditions(const HydraInflowProfile& inflow,
                                                         const HydraOutflowProfile& outflow,
                                                         const GasModel& gas,
                                                         Real mach);

}  // namespace nssolver
