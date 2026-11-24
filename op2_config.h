#pragma once

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct SolverConfig {
  std::string mesh_file;
  std::string output_file;
  std::string init_mode = "uniform";
  std::string hydra_inflow_xml;
  std::string hydra_outflow_xml;
  double hydra_benchmark_mach = 0.20;
  int iterations = 200;
  double cfl = 0.1;
  int progress_interval = 10;
  int second_order = 0;
  // Second order: 1 the edge-local limiter, 2 van Albada's, 3 van Albada's where
  // the pressure sensor exceeds limiter_sensor (shocks), 0 none.
  int limiter = 1;
  double limiter_sensor = 5.0e-4;
  int include_viscous = 0;
  int include_sa = 0;
  int sa_comp = 0;     // TMR's SA-comp: the compressibility correction
  int sa_edwards = 0;  // TMR's SA-noft2-Edwards
  double gamma = 1.4;
  double gas_constant = 287.05;
  double rho_floor = 1.0e-4;
  double p_floor = 100.0;
  double primitive[6] = {1.225, 120.0, 0.0, 0.0, 101325.0, 0.0};
  // Outlet ghost state under backflow: extrapolate (the interior state with
  // the free-stream pressure, as for outflow) or freestream.
  std::string outlet_backflow = "extrapolate";

  // Time marching: rk4, or implicit pseudo-time steps through OP2's solver
  // layer at a CFL from implicit_cfl, times implicit_cfl_growth each
  // iteration, up to implicit_cfl_max; the Jacobian reassembled every
  // jacobian_interval iterations.
  std::string time_scheme = "rk4";
  double implicit_cfl = 10.0;
  double implicit_cfl_max = 0.0;    // implicit_cfl when 0
  double implicit_cfl_growth = 1.0;
  double implicit_relax = 1.0;
  double update_limit = 0.0;  // > 0: at most this relative change of rho and p per node and iteration
  // The SA variable's reference in the scaled system: rho nu of the free
  // stream's nu~, at least sa_scale times its laminar nu (1 the old scaling).
  double sa_scale = 1000.0;
  int jacobian_interval = 1;

  // Dual time stepping (implicit, time_step > 0): physical_steps steps of
  // time_step, each solving R(q) + V (c0 q - c1 q^n + c2 q^n-1) / time_step = 0
  // by up to inner_iterations pseudo-time iterations (the CFL ramp restarted)
  // until the residual falls by inner_tolerance; second-order backward
  // differences (time_order 2) after a first-order first step. restart_file
  // starts from a solution's q, snapshot_interval writes one every that many
  // steps, and output_file's .history.csv has the wall force of each step.
  double time_step = 0.0;
  int physical_steps = 0;
  int inner_iterations = 20;
  double inner_tolerance = 1.0e-3;
  int time_order = 2;
  std::string restart_file;
  int snapshot_interval = 0;

  // The linear solves, as the OP2 implicit proxy (apps/c/implicit) takes them:
  // the solver layer's JSON (solve_config; OP_SOLVE_CONFIG over it), the
  // library's preconditioner or a par-loop one (pc: lib, bjacobi, neumann with
  // terms steps), the library's preconditioner from A or its components'
  // blocks alone (pmat: A, decoupled), and the operator assembled or by
  // differences of the residual (op: assembled, matfree).
  std::string solve_config;
  std::string pc = "lib";
  int terms = 2;
  std::string pmat = "A";
  std::string op = "assembled";

  // Every dump_interval-th implicit iteration (0: never) writes the linear
  // system it solved, as dump_prefix_<iteration>.h5 (matrix A, right-hand
  // side A/b, solution A/x: apps/c/solve_hdf5 solves it again), and a line
  // to dump_prefix_index.csv.
  int dump_interval = 0;
  std::string dump_prefix = "outputs-op2/system";

  // Run: partitioner (kway, random, inertial), renumbering each rank's mesh
  // (reorder), OP2's diagnostics level and timing output, and a check that
  // fails the run unless L2(rho) falls by the factor check.
  std::string partition = "kway";
  int reorder = 0;
  int diags = 2;
  int timing = 1;
  double check = 0.0;
};

// Sets the member a config key (or command-line option) names; false for a
// key no member has.
inline bool apply_config_value(SolverConfig &cfg, const std::string &key, const std::string &value) {
  if (key == "mesh_file") cfg.mesh_file = value;
  else if (key == "output_file") cfg.output_file = value;
  else if (key == "init_mode") cfg.init_mode = value;
  else if (key == "hydra_inflow_xml") cfg.hydra_inflow_xml = value;
  else if (key == "hydra_outflow_xml") cfg.hydra_outflow_xml = value;
  else if (key == "hydra_benchmark_mach") cfg.hydra_benchmark_mach = std::stod(value);
  else if (key == "iterations") cfg.iterations = std::stoi(value);
  else if (key == "cfl") cfg.cfl = std::stod(value);
  else if (key == "progress_interval") cfg.progress_interval = std::stoi(value);
  else if (key == "second_order") cfg.second_order = std::stoi(value);
  else if (key == "limiter") cfg.limiter = std::stoi(value);
  else if (key == "include_viscous") cfg.include_viscous = std::stoi(value);
  else if (key == "include_sa") cfg.include_sa = std::stoi(value);
  else if (key == "sa_comp") cfg.sa_comp = std::stoi(value);
  else if (key == "sa_edwards") cfg.sa_edwards = std::stoi(value);
  else if (key == "limiter_sensor") cfg.limiter_sensor = std::stod(value);
  else if (key == "gamma") cfg.gamma = std::stod(value);
  else if (key == "gas_constant") cfg.gas_constant = std::stod(value);
  else if (key == "rho_floor") cfg.rho_floor = std::stod(value);
  else if (key == "p_floor") cfg.p_floor = std::stod(value);
  else if (key == "rho") cfg.primitive[0] = std::stod(value);
  else if (key == "u") cfg.primitive[1] = std::stod(value);
  else if (key == "v") cfg.primitive[2] = std::stod(value);
  else if (key == "w") cfg.primitive[3] = std::stod(value);
  else if (key == "p") cfg.primitive[4] = std::stod(value);
  else if (key == "nu_tilde") cfg.primitive[5] = std::stod(value);
  else if (key == "outlet_backflow") cfg.outlet_backflow = value;
  else if (key == "time_scheme") cfg.time_scheme = value;
  else if (key == "implicit_cfl") cfg.implicit_cfl = std::stod(value);
  else if (key == "implicit_cfl_max") cfg.implicit_cfl_max = std::stod(value);
  else if (key == "implicit_cfl_growth") cfg.implicit_cfl_growth = std::stod(value);
  else if (key == "implicit_relax") cfg.implicit_relax = std::stod(value);
  else if (key == "update_limit") cfg.update_limit = std::stod(value);
  else if (key == "sa_scale") cfg.sa_scale = std::stod(value);
  else if (key == "jacobian_interval") cfg.jacobian_interval = std::stoi(value);
  else if (key == "solve_config") cfg.solve_config = value;
  else if (key == "time_step") cfg.time_step = std::stod(value);
  else if (key == "physical_steps") cfg.physical_steps = std::stoi(value);
  else if (key == "inner_iterations") cfg.inner_iterations = std::stoi(value);
  else if (key == "inner_tolerance") cfg.inner_tolerance = std::stod(value);
  else if (key == "time_order") cfg.time_order = std::stoi(value);
  else if (key == "restart_file") cfg.restart_file = value;
  else if (key == "snapshot_interval") cfg.snapshot_interval = std::stoi(value);
  else if (key == "pc") cfg.pc = value;
  else if (key == "terms") cfg.terms = std::stoi(value);
  else if (key == "pmat") cfg.pmat = value;
  else if (key == "op") cfg.op = value;
  else if (key == "dump_interval") cfg.dump_interval = std::stoi(value);
  else if (key == "dump_prefix") cfg.dump_prefix = value;
  else if (key == "partition") cfg.partition = value;
  else if (key == "reorder") cfg.reorder = std::stoi(value);
  else if (key == "diags") cfg.diags = std::stoi(value);
  else if (key == "timing") cfg.timing = std::stoi(value);
  else if (key == "check") cfg.check = std::stod(value);
  else return false;
  return true;
}

inline void validate_config(const SolverConfig &cfg) {
  if (cfg.mesh_file.empty()) throw std::runtime_error("config missing mesh_file");
  if (cfg.output_file.empty()) throw std::runtime_error("config missing output_file");
  auto one_of = [](const std::string &key, const std::string &value, std::initializer_list<const char *> allowed) {
    for (const char *a : allowed)
      if (value == a) return;
    throw std::runtime_error("config: " + key + " = " + value + " is not one of the choices");
  };
  one_of("outlet_backflow", cfg.outlet_backflow, {"extrapolate", "freestream"});
  one_of("time_scheme", cfg.time_scheme, {"rk4", "implicit"});
  one_of("pc", cfg.pc, {"lib", "bjacobi", "neumann"});
  one_of("pmat", cfg.pmat, {"A", "decoupled"});
  one_of("op", cfg.op, {"assembled", "matfree"});
  one_of("partition", cfg.partition, {"kway", "random", "inertial"});
  if (cfg.dump_interval < 0) throw std::runtime_error("config: dump_interval must not be negative");
  if (cfg.jacobian_interval < 1) throw std::runtime_error("config: jacobian_interval must be at least 1");
  if (cfg.implicit_cfl <= 0.0) throw std::runtime_error("config: implicit_cfl must be positive");
  if (cfg.time_step > 0.0 && cfg.time_scheme != "implicit")
    throw std::runtime_error("config: time_step needs time_scheme = implicit");
  if (cfg.time_order != 1 && cfg.time_order != 2) throw std::runtime_error("config: time_order must be 1 or 2");
}

inline std::string trim_copy(std::string value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

inline void load_config_file(const std::string &path, SolverConfig &cfg) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("failed to open config file: " + path);

  std::string line;
  while (std::getline(file, line)) {
    line = trim_copy(line);
    if (line.empty() || line[0] == '#') continue;
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = trim_copy(line.substr(0, eq));
    const std::string value = trim_copy(line.substr(eq + 1));
    apply_config_value(cfg, key, value);
  }
}

inline std::vector<std::vector<double>> read_hydra_xml_rows(const std::string &path, std::size_t expected_columns) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("failed to open Hydra XML: " + path);
  std::ostringstream text;
  text << file.rdbuf();
  const std::string content = text.str();
  const std::size_t begin = content.find("<data>");
  const std::size_t end = content.find("</data>");
  if (begin == std::string::npos || end == std::string::npos || end <= begin) {
    throw std::runtime_error("Hydra XML missing <data> block: " + path);
  }

  std::istringstream lines(content.substr(begin + 6, end - begin - 6));
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream values(line);
    std::vector<double> row;
    double x = 0.0;
    while (values >> x) row.push_back(x);
    if (row.empty()) continue;
    if (row.size() != expected_columns) {
      throw std::runtime_error("unexpected Hydra XML column count in " + path);
    }
    rows.push_back(row);
  }
  if (rows.empty()) throw std::runtime_error("Hydra XML has no data rows: " + path);
  return rows;
}

inline double mean_column(const std::vector<std::vector<double>> &rows, std::size_t column) {
  double sum = 0.0;
  for (const auto &row : rows) sum += row[column];
  return sum / static_cast<double>(rows.size());
}

inline void derive_hydra_benchmark_primitive(SolverConfig &cfg) {
  const auto inflow = read_hydra_xml_rows(cfg.hydra_inflow_xml, 6);
  const auto outflow = read_hydra_xml_rows(cfg.hydra_outflow_xml, 2);
  const double tt = mean_column(inflow, 1);
  const double pt = mean_column(inflow, 2);
  const double nu = mean_column(inflow, 5);
  const double pexit = mean_column(outflow, 1);
  const double mach = cfg.hydra_benchmark_mach;

  const double tr = 1.0 + 0.5 * (cfg.gamma - 1.0) * mach * mach;
  const double t = tt / tr;
  const double p = pt / std::pow(tr, cfg.gamma / (cfg.gamma - 1.0));
  const double rho = p / (cfg.gas_constant * t);
  const double a = std::sqrt(cfg.gamma * cfg.gas_constant * t);
  const double u = mach * a;

  cfg.primitive[0] = rho;
  cfg.primitive[1] = u;
  cfg.primitive[2] = 0.0;
  cfg.primitive[3] = 0.0;
  cfg.primitive[4] = p;
  cfg.primitive[5] = nu;

  std::cout << "[benchmark] mach=" << mach
            << ", Tt_in=" << tt
            << ", Pt_in=" << pt
            << ", p_static_benchmark=" << p
            << ", p_exit_original_mean=" << pexit << "\n";
  std::cout << "[benchmark] original exit pressure rise is not imposed in the OP2 baseline; "
               "this run is a non-rotating matched-pressure benchmark\n";
}
