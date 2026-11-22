#pragma once

#include <array>
#include <limits>
#include <string>
#include <vector>

#include "nssolver/types.hpp"

namespace nssolver {

// Farfield and Inlet impose the free stream strongly at their nodes in viscous
// runs; Characteristic (a Riemann far field) and TotalInlet (total pressure
// and temperature in, one quantity from the interior) only through the flux.
enum class BoundaryType {
    Farfield,
    Inlet,
    Outlet,
    SlipWall,
    NoSlipWall,
    Characteristic,
    TotalInlet
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
    std::vector<Index> num_nodes;
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

inline std::array<Index, 4> boundary_face_nodes(const BoundaryFaces& faces, std::size_t f) {
    return {faces.n1[f], faces.n2[f], faces.n3[f], faces.n4[f]};
}

inline std::size_t boundary_face_num_nodes(const BoundaryFaces& faces, std::size_t f) {
    return faces.num_nodes.size() == faces.count ? static_cast<std::size_t>(faces.num_nodes[f]) : 4U;
}

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

// A 2D structured grid from a formatted PLOT3D file (one block, x-y plane,
// as the NASA Turbulence Modeling Resource gives them), extruded one cell in
// z between slip planes: i along the jmin boundary, which is a no-slip wall
// where the face's x lies in [wall_x0, wall_x1] and lower_other elsewhere.
struct Plot3D2DSpec {
    std::string path;
    Real lz {0.05};
    Real wall_x0 {0.0};
    Real wall_x1 {std::numeric_limits<Real>::max()};
    BoundaryType inlet {BoundaryType::Inlet};
    BoundaryType outlet {BoundaryType::Outlet};
    BoundaryType upper {BoundaryType::Farfield};
    BoundaryType lower_other {BoundaryType::SlipWall};
};

// A circular cylinder's O-grid, extruded one layer in z between slip planes:
// ntheta points around, nr from the wall (first_spacing off it) to
// farfield_radius.
struct CylinderSpec {
    Index ntheta {192};
    Index nr {128};
    Index nz {2};
    Real radius {0.5};
    Real farfield_radius {30.0};
    Real first_spacing {2.0e-3};
    Real lz {0.1};
    BoundaryType wall {BoundaryType::NoSlipWall};
    BoundaryType farfield {BoundaryType::Characteristic};
};

// Planar multi-block grids in PLOT3D (Fortran unformatted, x-y, nk = 1, as
// the AIAA buffet workshop's OAT15A grids), extruded one layer between slip
// planes, blocks joined where their points coincide (within merge_tolerance,
// after scale): no-slip walls where a face's centre lies within x in
// [wall_x0, wall_x1], |y| <= wall_y, far field elsewhere.
struct Plot3DBlocksSpec {
    std::string path;
    Real scale {1.0};
    Real lz {0.01};
    Real merge_tolerance {1.0e-9};
    Real wall_x0 {-0.01};
    Real wall_x1 {1.01};
    Real wall_y {0.2};
    BoundaryType farfield {BoundaryType::Characteristic};
};

// An O-grid about a biconvex circular-arc section of unit chord and the given
// thickness: surface_points per side (clustered at both edges), normal_points
// out to farfield_radius from first_spacing.
struct CircularArcSpec {
    Real thickness {0.18};
    Index surface_points {161};
    Index normal_points {129};
    Real first_spacing {2.0e-6};
    Real farfield_radius {50.0};
    Real blend_length {0.05};
    Real lz {0.01};
    BoundaryType farfield {BoundaryType::Characteristic};
};

Mesh make_structured_box_mesh(const StructuredBoxSpec& spec);
Mesh make_bump_channel_mesh(const BumpChannelSpec& spec);
Mesh make_flat_plate_mesh(const FlatPlateSpec& spec);
Mesh make_bump_3d_mesh(const Bump3DSpec& spec);
Mesh make_axisymmetric_body_mesh(const AxisymmetricBodySpec& spec);
Mesh make_naca_finite_wing_mesh(const NacaWingSpec& spec);
Mesh make_plot3d_2d_mesh(const Plot3D2DSpec& spec);
Mesh make_cylinder_mesh(const CylinderSpec& spec);
Mesh make_plot3d_blocks_mesh(const Plot3DBlocksSpec& spec);
Mesh make_circular_arc_mesh(const CircularArcSpec& spec);
Real total_nodal_volume(const Mesh& mesh);
// Each boundary face's nodes' shares of it, 4 area vectors per face (unused
// ones zero): the median sub-face (node, edge midpoints, face centroid) that
// closes the node's control volume.
std::vector<Real> boundary_node_normals(const Mesh& mesh);

}  // namespace nssolver
