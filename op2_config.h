#pragma once

#include <cmath>
#include <fstream>
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
  int include_viscous = 0;
  int include_sa = 0;
  double gamma = 1.4;
  double gas_constant = 287.05;
  double rho_floor = 1.0e-4;
  double p_floor = 100.0;
  double primitive[6] = {1.225, 120.0, 0.0, 0.0, 101325.0, 0.0};
};

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
    else if (key == "include_viscous") cfg.include_viscous = std::stoi(value);
    else if (key == "include_sa") cfg.include_sa = std::stoi(value);
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
  }

  if (cfg.mesh_file.empty()) throw std::runtime_error("config missing mesh_file");
  if (cfg.output_file.empty()) throw std::runtime_error("config missing output_file");
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
