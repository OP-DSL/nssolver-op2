#include "nssolver/hydra_benchmark.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nssolver {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open Hydra boundary profile: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<std::vector<Real>> read_data_rows(const std::string& path, std::size_t expected_columns) {
    const std::string text = read_file(path);
    const std::string open_tag = "<data>";
    const std::string close_tag = "</data>";
    const std::size_t begin = text.find(open_tag);
    const std::size_t end = text.find(close_tag);
    if (begin == std::string::npos || end == std::string::npos || end <= begin) {
        throw std::runtime_error("Hydra boundary profile has no <data> block: " + path);
    }

    std::istringstream lines(text.substr(begin + open_tag.size(), end - begin - open_tag.size()));
    std::vector<std::vector<Real>> rows;
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream values(line);
        std::vector<Real> row;
        Real value = 0.0;
        while (values >> value) {
            row.push_back(value);
        }
        if (row.empty()) {
            continue;
        }
        if (row.size() != expected_columns) {
            throw std::runtime_error("Hydra boundary profile row has unexpected column count: " + path);
        }
        rows.push_back(row);
    }
    if (rows.empty()) {
        throw std::runtime_error("Hydra boundary profile has an empty <data> block: " + path);
    }
    return rows;
}

Real mean_column(const std::vector<std::vector<Real>>& rows, std::size_t column) {
    Real sum = 0.0;
    for (const auto& row : rows) {
        sum += row[column];
    }
    return sum / static_cast<Real>(rows.size());
}

}  // namespace

HydraInflowProfile read_hydra_inflow_profile(const std::string& path) {
    const auto rows = read_data_rows(path, 6);
    return {
        .rows = rows.size(),
        .mean_radius = mean_column(rows, 0),
        .total_temperature = mean_column(rows, 1),
        .total_pressure = mean_column(rows, 2),
        .whirl_angle = mean_column(rows, 3),
        .pitch_angle = mean_column(rows, 4),
        .nu_tilde = mean_column(rows, 5),
    };
}

HydraOutflowProfile read_hydra_outflow_profile(const std::string& path) {
    const auto rows = read_data_rows(path, 2);
    return {
        .rows = rows.size(),
        .mean_radius = mean_column(rows, 0),
        .static_pressure = mean_column(rows, 1),
    };
}

HydraBenchmarkConditions make_hydra_benchmark_conditions(const HydraInflowProfile& inflow,
                                                         const HydraOutflowProfile& outflow,
                                                         const GasModel& gas,
                                                         Real mach) {
    if (mach <= 0.0) {
        throw std::runtime_error("Hydra benchmark Mach number must be positive");
    }
    if (inflow.total_temperature <= 0.0 || inflow.total_pressure <= 0.0) {
        throw std::runtime_error("Hydra inflow total conditions must be positive");
    }

    const Real temperature_ratio = 1.0 + 0.5 * (gas.gamma - 1.0) * mach * mach;
    const Real static_temperature = inflow.total_temperature / temperature_ratio;
    const Real static_pressure = inflow.total_pressure / std::pow(temperature_ratio, gas.gamma / (gas.gamma - 1.0));
    const Real density = static_pressure / (gas.gas_constant * static_temperature);
    const Real speed = mach * std::sqrt(gas.gamma * gas.gas_constant * static_temperature);

    return {
        .primitive = Primitive {
            .rho = density,
            .u = speed,
            .v = 0.0,
            .w = 0.0,
            .p = static_pressure,
            .nu_tilde = inflow.nu_tilde,
        },
        .mach = mach,
        .inlet_total_temperature = inflow.total_temperature,
        .inlet_total_pressure = inflow.total_pressure,
        .original_exit_static_pressure = outflow.static_pressure,
    };
}

}  // namespace nssolver
