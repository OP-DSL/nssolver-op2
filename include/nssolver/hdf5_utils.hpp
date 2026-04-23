#pragma once

#ifdef NSSOLVER_HAVE_HDF5

#include <hdf5.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nssolver::hdf5 {

class Handle {
public:
    Handle() = default;
    Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&& other) noexcept : id_(other.id_), closer_(other.closer_) {
        other.id_ = -1;
        other.closer_ = nullptr;
    }

    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = other.id_;
            closer_ = other.closer_;
            other.id_ = -1;
            other.closer_ = nullptr;
        }
        return *this;
    }

    ~Handle() {
        reset();
    }

    hid_t get() const {
        return id_;
    }

private:
    void reset() {
        if (id_ >= 0 && closer_ != nullptr) {
            closer_(id_);
        }
        id_ = -1;
        closer_ = nullptr;
    }

    hid_t id_ {-1};
    herr_t (*closer_)(hid_t) {nullptr};
};

inline void require(herr_t status, const std::string& context) {
    if (status < 0) {
        throw std::runtime_error(context);
    }
}

inline hid_t require_id(hid_t id, const std::string& context) {
    if (id < 0) {
        throw std::runtime_error(context);
    }
    return id;
}

inline Handle open_file_readonly(const std::string& path) {
    return Handle(require_id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "failed to open HDF5 file: " + path),
                  &H5Fclose);
}

inline Handle create_file_truncate(const std::string& path) {
    return Handle(require_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                             "failed to create HDF5 file: " + path),
                  &H5Fclose);
}

inline bool dataset_exists(const Handle& file, const std::string& name) {
    return H5Lexists(file.get(), name.c_str(), H5P_DEFAULT) > 0;
}

inline Handle open_dataset(const Handle& file, const std::string& name) {
    return Handle(require_id(H5Dopen2(file.get(), name.c_str(), H5P_DEFAULT), "failed to open dataset: " + name),
                  &H5Dclose);
}

inline std::vector<hsize_t> get_dims(const Handle& dataset) {
    Handle space(require_id(H5Dget_space(dataset.get()), "failed to get dataset dataspace"), &H5Sclose);
    const int rank = H5Sget_simple_extent_ndims(space.get());
    if (rank < 0) {
        throw std::runtime_error("failed to read dataset rank");
    }
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    require(H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr), "failed to read dataset dimensions");
    return dims;
}

template <typename T>
hid_t native_type() {
    if constexpr (std::is_same_v<T, double>) {
        return H5T_NATIVE_DOUBLE;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return H5T_NATIVE_INT32;
    } else if constexpr (std::is_same_v<T, int>) {
        return H5T_NATIVE_INT;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        return H5T_NATIVE_UCHAR;
    } else {
        static_assert(!sizeof(T), "unsupported HDF5 native type");
    }
}

template <typename T>
std::pair<std::vector<hsize_t>, std::vector<T>> read_dataset(const Handle& file, const std::string& name) {
    const Handle dataset = open_dataset(file, name);
    const std::vector<hsize_t> dims = get_dims(dataset);
    std::size_t count = 1;
    for (hsize_t dim : dims) {
        count *= static_cast<std::size_t>(dim);
    }
    std::vector<T> values(count);
    require(H5Dread(dataset.get(), native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
            "failed to read dataset: " + name);
    return {dims, values};
}

template <typename T>
void write_dataset(const Handle& file, const std::string& name, const std::vector<hsize_t>& dims, const T* data) {
    Handle space(require_id(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr),
                            "failed to create dataspace for dataset: " + name),
                 &H5Sclose);
    Handle dataset(require_id(H5Dcreate2(file.get(), name.c_str(), native_type<T>(), space.get(), H5P_DEFAULT,
                                         H5P_DEFAULT, H5P_DEFAULT),
                              "failed to create dataset: " + name),
                   &H5Dclose);
    require(H5Dwrite(dataset.get(), native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
            "failed to write dataset: " + name);
}

}  // namespace nssolver::hdf5

#endif
