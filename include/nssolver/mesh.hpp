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

Mesh make_structured_box_mesh(const StructuredBoxSpec& spec);
Mesh make_bump_channel_mesh(const BumpChannelSpec& spec);
Mesh make_flat_plate_mesh(const FlatPlateSpec& spec);
Real total_nodal_volume(const Mesh& mesh);

}  // namespace nssolver
