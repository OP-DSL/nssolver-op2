#pragma once

#include <array>
#include <string>
#include <vector>

#include "nssolver/types.hpp"

namespace nssolver {

enum class BoundaryType {
    Farfield,
    Inlet,
    Outlet,
    SlipWall,
    NoSlipWall
};

struct Nodes {
    std::size_t count {};
    std::vector<Real> x, y, z;
    std::vector<Real> vol;
    std::vector<Real> wall_dist;
};

struct Edges {
    std::size_t count {};
    std::vector<Index> node_L, node_R;
    std::vector<Real> nx, ny, nz;
    std::vector<Real> area;
};

struct BoundaryFaces {
    std::size_t count {};
    std::vector<Index> n1, n2, n3, n4;
    std::vector<Real> nx, ny, nz;
    std::vector<Real> area;
    std::vector<Index> group_id;
    std::vector<BoundaryType> type;
    std::vector<std::string> name;
};

struct Mesh {
    Nodes nodes;
    Edges edges;
    BoundaryFaces boundary_faces;
    std::vector<std::vector<Index>> node_to_edges;
    std::vector<std::vector<Index>> node_to_boundary_faces;
};

struct StructuredBoxSpec {
    Index nx {5};
    Index ny {5};
    Index nz {2};
    Real lx {1.0};
    Real ly {1.0};
    Real lz {0.1};
    BoundaryType xmin {BoundaryType::Farfield};
    BoundaryType xmax {BoundaryType::Farfield};
    BoundaryType ymin {BoundaryType::SlipWall};
    BoundaryType ymax {BoundaryType::SlipWall};
    BoundaryType zmin {BoundaryType::SlipWall};
    BoundaryType zmax {BoundaryType::SlipWall};
};

struct BumpChannelSpec {
    Index nx {41};
    Index ny {25};
    Index nz {2};
    Real lx {3.0};
    Real ly {1.0};
    Real lz {0.05};
    Real bump_center {1.5};
    Real bump_half_width {0.4};
    Real bump_height {0.08};
    BoundaryType inlet {BoundaryType::Farfield};
    BoundaryType outlet {BoundaryType::Farfield};
    BoundaryType lower_wall {BoundaryType::SlipWall};
    BoundaryType upper_wall {BoundaryType::SlipWall};
};

struct FlatPlateSpec {
    Index nx {61};
    Index ny {41};
    Index nz {2};
    Real lx {1.5};
    Real ly {0.6};
    Real lz {0.02};
    Real leading_edge_x {0.2};
    Real wall_normal_growth {2.5};
    BoundaryType inlet {BoundaryType::Farfield};
    BoundaryType outlet {BoundaryType::Farfield};
    BoundaryType plate_type {BoundaryType::NoSlipWall};
    BoundaryType lower_upstream_type {BoundaryType::SlipWall};
    BoundaryType upper_type {BoundaryType::Farfield};
};

struct Bump3DSpec {
    Index nx {81};
    Index ny {33};
    Index nz {41};
    Real lx {3.0};
    Real ly {1.0};
    Real lz {1.0};
    Real bump_center_x {1.5};
    Real bump_half_width_x {0.5};
    Real bump_center_z {0.5};
    Real bump_half_width_z {0.2};
    Real bump_height {0.08};
    BoundaryType inlet {BoundaryType::Farfield};
    BoundaryType outlet {BoundaryType::Farfield};
    BoundaryType lower_wall {BoundaryType::SlipWall};
    BoundaryType upper_wall {BoundaryType::SlipWall};
    BoundaryType side_wall {BoundaryType::SlipWall};
};

struct AxisymmetricBodySpec {
    Index nx {121};
    Index nr {33};
    Index ntheta {17};
    Real lx {3.0};
    Real body_start_x {0.4};
    Real body_length {2.2};
    Real body_radius_max {0.18};
    Real body_tail_radius {0.015};
    Real farfield_radius {0.9};
    Real wedge_angle_degrees {20.0};
    Real radial_growth {3.0};
    BoundaryType inlet {BoundaryType::Farfield};
    BoundaryType outlet {BoundaryType::Farfield};
    BoundaryType body_wall {BoundaryType::SlipWall};
    BoundaryType farfield {BoundaryType::Farfield};
    BoundaryType wedge_wall {BoundaryType::SlipWall};
};

struct NacaWingSpec {
    Index nx {121};
    Index ny {41};
    Index nz {81};
    Real lx {2.8};
    Real ly {0.9};
    Real lz {1.6};
    Real wing_origin_x {0.45};
    Real wing_center_z {0.8};
    Real span {1.0};
    Real root_chord {0.8};
    Real tip_chord {0.4};
    Real sweep_degrees {20.0};
    Real thickness_ratio {0.12};
    Real camber_ratio {0.0};
    Real camber_position {0.4};
    Real vertical_offset {0.0};
    Real wall_normal_growth {4.0};
    BoundaryType inlet {BoundaryType::Farfield};
    BoundaryType outlet {BoundaryType::Farfield};
    BoundaryType wing_wall {BoundaryType::NoSlipWall};
    BoundaryType lower_farfield {BoundaryType::SlipWall};
    BoundaryType upper_farfield {BoundaryType::Farfield};
    BoundaryType side_farfield {BoundaryType::Farfield};
};

Mesh make_structured_box_mesh(const StructuredBoxSpec& spec);
Mesh make_bump_channel_mesh(const BumpChannelSpec& spec);
Mesh make_flat_plate_mesh(const FlatPlateSpec& spec);
Mesh make_bump_3d_mesh(const Bump3DSpec& spec);
Mesh make_axisymmetric_body_mesh(const AxisymmetricBodySpec& spec);
Mesh make_naca_finite_wing_mesh(const NacaWingSpec& spec);
Real total_nodal_volume(const Mesh& mesh);

}  // namespace nssolver
